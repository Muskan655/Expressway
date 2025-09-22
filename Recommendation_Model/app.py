import io
import os
import uuid
import datetime
import logging
from typing import List, Tuple

import requests
import numpy as np
import pandas as pd
from PIL import Image
import colorsys

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util

# ------------------------- Config -------------------------
DATA_CSV = "expressway_dataset.csv"
EMBED_FILE = "clip_catalog_cache.pt"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("expressway")

# ------------------------- App -------------------------
app = FastAPI(title="Expressway Stylist (CLIP + Color + NLP)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------- Load dataset -------------------------
df = pd.read_csv(DATA_CSV).fillna("")
df["id"] = df["id"].astype(int)
df["category"] = df["category"].astype(str).str.strip().str.lower()
df["occasion"] = df["occasion"].astype(str).str.strip().str.lower()
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"].fillna(0), errors="coerce").fillna(0).astype(int)

# ------------------------- Models -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Using device: %s", device)

log.info("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

log.info("Loading semantic text model (sentence-transformers)...")
nlp_model = SentenceTransformer("all-MiniLM-L6-v2")

ALL_CATEGORIES = np.array(sorted(df["category"].unique()))
cat_embs = nlp_model.encode(ALL_CATEGORIES.tolist(), convert_to_tensor=True) if len(ALL_CATEGORIES) else None

# ------------------------- Helpers: image download & color -------------------------
def fetch_image_from_source(src: str, timeout=6) -> Image.Image:
    try:
        if str(src).lower().startswith("http"):
            headers = {"User-Agent": USER_AGENT}
            r = requests.get(src, headers=headers, timeout=timeout)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        else:
            return Image.open(src).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load image {src}: {e}")

def get_dominant_color(img: Image.Image, resize=64) -> Tuple[int,int,int]:
    try:
        small = img.resize((resize, resize))
        colors = small.getcolors(maxcolors=resize*resize)
        if colors:
            colors.sort(reverse=True, key=lambda x: x[0])
            return tuple(colors[0][1][:3])
        arr = np.array(small).reshape(-1, 3)
        return tuple(np.round(arr.mean(axis=0)).astype(int).tolist())
    except Exception:
        arr = np.array(img.resize((32,32))).reshape(-1,3)
        return tuple(np.round(arr.mean(axis=0)).astype(int).tolist())

def rgb_to_rel_luminance(rgb: Tuple[int,int,int]) -> float:
    def channel(c):
        x = c/255.0
        return x/12.92 if x <= 0.03928 else ((x+0.055)/1.055) ** 2.4
    r,g,b = rgb
    return 0.2126*channel(r) + 0.7152*channel(g) + 0.0722*channel(b)

def color_contrast_score(rgb1: Tuple[int,int,int], rgb2: Tuple[int,int,int]) -> float:
    l1 = rgb_to_rel_luminance(rgb1)
    l2 = rgb_to_rel_luminance(rgb2)
    lum_diff = abs(l1 - l2)
    h1,_,_ = colorsys.rgb_to_hls(rgb1[0]/255, rgb1[1]/255, rgb1[2]/255)
    h2,_,_ = colorsys.rgb_to_hls(rgb2[0]/255, rgb2[1]/255, rgb2[2]/255)
    hue_diff = min(abs(h1-h2), 1-abs(h1-h2))*2
    score = 0.55*lum_diff + 0.45*hue_diff
    return max(0.0, min(1.0, score))

# ------------------------- Precompute CLIP embeddings + colors -------------------------
if os.path.exists(EMBED_FILE):
    try:
        cache = torch.load(EMBED_FILE, map_location=device)
        catalog_embeddings = cache["embeddings"].to(device)
        catalog_colors = cache["colors"]
        if len(catalog_colors) != len(df):
            raise RuntimeError("Mismatch")
        log.info("Loaded cached embeddings and colors.")
    except:
        os.remove(EMBED_FILE)
        catalog_embeddings, catalog_colors = None, None
else:
    catalog_embeddings, catalog_colors = None, None

if catalog_embeddings is None:
    log.info("Computing CLIP embeddings + colors...")
    embs, colors = [], []
    for i,row in df.iterrows():
        img = None
        try:
            img = fetch_image_from_source(row.get("image_url",""))
        except Exception as e:
            log.warning("Image load failed: %s", e)
        if img:
            try:
                inputs = clip_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    emb = clip_model.get_image_features(**inputs)
                emb = emb.squeeze(0)/(emb.norm(p=2)+1e-10)
                embs.append(emb.cpu())
            except:
                embs.append(torch.zeros(512))
            try:
                colors.append(tuple(int(x) for x in get_dominant_color(img)))
            except:
                colors.append((128,128,128))
        else:
            embs.append(torch.zeros(512))
            colors.append((128,128,128))
    catalog_embeddings = torch.stack(embs).to(device)
    catalog_colors = colors
    torch.save({"embeddings": catalog_embeddings.cpu(), "colors": catalog_colors}, EMBED_FILE)

# ------------------------- Embedding helpers -------------------------
def embed_uploaded_image_bytes(contents: bytes):
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb.squeeze(0)/(emb.norm(p=2)+1e-10)
    dom_color = get_dominant_color(img)
    return emb, dom_color

def get_top_matches_from_emb(query_emb: torch.Tensor, candidate_indices: List[int], top_k: int = 3):
    if not candidate_indices:
        return []
    cand_embs = catalog_embeddings[candidate_indices]
    q = query_emb.view(-1,1).to(device)
    sims = (cand_embs @ q).squeeze(1).cpu().numpy()
    order = np.argsort(sims)[::-1][:top_k]
    return [(candidate_indices[int(idx)], float(sims[int(idx)])) for idx in order]

# ------------------------- NLP prompt -> categories -------------------------
def parse_prompt_to_categories(prompt: str, top_k: int = 3, threshold: float = 0.35) -> List[str]:
    if not prompt or len(ALL_CATEGORIES)==0:
        return []
    p_emb = nlp_model.encode(prompt, convert_to_tensor=True)
    sims = util.cos_sim(p_emb, cat_embs)[0].cpu().numpy().tolist()
    pairs = sorted([(ALL_CATEGORIES[i], sims[i]) for i in range(len(ALL_CATEGORIES))], key=lambda x:x[1], reverse=True)
    chosen = [c for c,score in pairs[:top_k] if score>=threshold]
    if not chosen and pairs:
        chosen = [pairs[0][0]]
    return [str(x).lower() for x in chosen]



# ------------------------- Config / Imports -------------------------
import uuid
import pandas as pd
# ... other imports ...

# ------------------------- Occasion normalization -------------------------
OCCASION_MAP = {
    "formal": ["meeting", "interview", "presentation", "office", "business"],
    "casual": ["trip", "outing", "college", "friends", "hangout", "shopping"],
    "party": ["party", "birthday", "function", "wedding", "festival", "celebration"],
    "sports": ["gym", "running", "yoga", "workout", "exercise"]
}

def normalize_occasion(raw: str) -> str:
    """Map raw occasion string to normalized label (formal, casual, party, sports)."""
    if not raw:
        return ""
    raw = raw.strip().lower()
    for norm, keywords in OCCASION_MAP.items():
        for kw in keywords:
            if kw in raw:
                return norm
    return raw  # fallback

# ------------------------- Event ID generator -------------------------
def gen_event_id():
    return str(uuid.uuid4())[:8]


# ------------------------- Outfit slots -------------------------
OUTFIT_SLOTS = {
    "wedding": ["saree", "jewellery", "heels"],
    "casual": ["jeans", "tshirts", "sneakers"],
    "formal": ["shirts", "trousers", "shoes"]
}

# ------------------------- Core: build outfit -------------------------
import random
from sentence_transformers import util

def build_outfit(event: dict, num_outfits: int = 3, per_slot_k: int = 3):
    event_occ = (event.get("occasion") or "").strip().lower()
    if not event_occ:
        return []

    # 1. Try exact match first
    strict_df = df[df["occasion"].str.lower() == event_occ].copy()

    # 2. If no exact match, fallback to nearest occasion using embeddings
    if strict_df.empty:
        occ_candidates = df["occasion"].unique().tolist()
        occ_emb = nlp_model.encode(event_occ, convert_to_tensor=True)
        occ_embs = nlp_model.encode(occ_candidates, convert_to_tensor=True)
        sim_scores = util.cos_sim(occ_emb, occ_embs)[0].cpu().numpy()
        best_idx = int(sim_scores.argmax())
        best_match = occ_candidates[best_idx]
        strict_df = df[df["occasion"] == best_match].copy()

    if strict_df.empty:
        return []  # no items even after fallback

    # Get relevant slots for this occasion
    slots = OUTFIT_SLOTS.get(event_occ) or list(strict_df["category"].value_counts().index[:3])
    base_slot = slots[0] if slots else None

    # Base candidates
    base_candidates = strict_df[strict_df["category"] == base_slot] if base_slot else strict_df
    if base_candidates.empty:
        return []
    base_candidates = base_candidates.head(50)

    outfits = []
    used_complements = {slot: set() for slot in slots}
    used_bases = 0
    shuffled_base_indices = base_candidates.index.tolist()
    random.shuffle(shuffled_base_indices)

    for idx in shuffled_base_indices:
        if used_bases >= num_outfits:
            break

        base_row = base_candidates.loc[idx]
        base_index = int(idx)
        base_emb = catalog_embeddings[base_index].unsqueeze(0)
        outfit = {"base": base_row.to_dict(), "complements": {}}

        for slot in slots:
            if slot == base_row["category"]:
                continue

            # Complement items strictly from the same occasion
            slot_candidates = strict_df[strict_df["category"] == slot]
            if slot_candidates.empty:
                outfit["complements"][slot] = None
                continue

            candidate_indices = [i for i in slot_candidates.index.astype(int) if i not in used_complements[slot]]
            if not candidate_indices:
                outfit["complements"][slot] = None
                continue

            matches = get_top_matches_from_emb(base_emb.squeeze(0), candidate_indices, top_k=per_slot_k)
            if matches:
                chosen_idx, sim = random.choice(matches)
                outfit["complements"][slot] = slot_candidates.loc[chosen_idx].to_dict()
                outfit["complements"][slot]["_sim"] = sim
                used_complements[slot].add(chosen_idx)
            else:
                outfit["complements"][slot] = None

        outfits.append(outfit)
        used_bases += 1

    return outfits

# ------------------------- Core: wardrobe matcher -------------------------
def wardrobe_matcher(
    wardrobe_image_bytes: bytes = None,
    base_item_id: int = None,
    user_prompt: str = "",
    occasion: str = None,
    top_k_per_slot: int = 3,
    weights: dict = None
):
    """
    Wardrobe matcher:
    - Use uploaded image or base_item_id as query.
    - user_prompt: free-text like "shoes and trousers"
    - occasion: optional; normalized internally
    - Returns top items per category with visual + color + occasion scoring.
    """

    if weights is None:
        weights = {"visual": 0.6, "color": 0.3, "occasion": 0.1}

    # 1) Query embedding & base color
    if wardrobe_image_bytes:
        query_emb, base_color = embed_uploaded_image_bytes(wardrobe_image_bytes)
    elif base_item_id is not None:
        base_row = df[df["id"] == int(base_item_id)]
        if base_row.empty:
            return {"error": f"item id {base_item_id} not found"}
        base_idx = int(base_row.index[0])
        query_emb = catalog_embeddings[base_idx].unsqueeze(0)
        base_color = tuple(catalog_colors[base_idx]) if catalog_colors and base_idx < len(catalog_colors) else (128,128,128)
    else:
        return {"error": "must provide wardrobe_image_bytes or base_item_id"}

    # 2) Desired categories from prompt
    desired_cats = parse_prompt_to_categories(user_prompt, top_k=5, threshold=0.3)
    desired_cats = [c for c in desired_cats if c in df["category"].unique()]
    if not desired_cats:
        return {"error": "Could not infer any categories from prompt. Try 'shoes', 'trousers', 'heels' etc."}

    # 3) Normalize occasion
    norm_occ = normalize_occasion(occasion or "")
    
    # 4) Filter catalog by normalized occasion (strict)
    if norm_occ:
        search_df = df[df["occasion"].apply(lambda x: normalize_occasion(x) == norm_occ)].copy()
        if search_df.empty:
            # fallback: bias by closest occasion using NLP similarity
            occ_candidates = df["occasion"].unique().tolist()
            occ_emb = nlp_model.encode(norm_occ, convert_to_tensor=True)
            occ_embs = nlp_model.encode([normalize_occasion(o) for o in occ_candidates], convert_to_tensor=True)
            sim_scores = util.cos_sim(occ_emb, occ_embs)[0].cpu().numpy()
            best_idx = int(sim_scores.argmax())
            best_match = occ_candidates[best_idx]
            # bias: matched occasion first
            search_df = pd.concat([
                df[df["occasion"] == best_match],
                df[df["occasion"] != best_match]
            ]).drop_duplicates()
    else:
        search_df = df.copy()

    results = {}
    # 5) Compute top matches per category
    for cat in desired_cats:
        slot_df = search_df[search_df["category"] == cat]
        if slot_df.empty:
            results[cat] = []
            continue

        candidate_indices = list(slot_df.index.astype(int).tolist())
        matches = get_top_matches_from_emb(query_emb.squeeze(0), candidate_indices, top_k=len(candidate_indices))
        
        scored = []
        for idx, visual_sim in matches:
            idx = int(idx)
            cand_color = tuple(catalog_colors[idx]) if catalog_colors and idx < len(catalog_colors) else (128,128,128)
            cscore = color_contrast_score(base_color, cand_color)
            occ_score = 1.0 if (norm_occ and normalize_occasion(df.loc[idx, "occasion"]) == norm_occ) else 0.0
            combined = weights["visual"]*visual_sim + weights["color"]*cscore + weights["occasion"]*occ_score
            scored.append((idx, visual_sim, cscore, occ_score, combined))

        # sort by combined score and take top_k
        scored.sort(key=lambda x: x[4], reverse=True)
        top = scored[:top_k_per_slot]

        out_items = []
        for idx, vs, cs, oscore, comb in top:
            row = df.loc[int(idx)].to_dict()
            row["_visual_sim"] = vs
            row["_color_score"] = cs
            row["_occasion_score"] = oscore
            row["_combined_score"] = comb
            row["_dominant_color"] = catalog_colors[idx] if (catalog_colors and idx < len(catalog_colors)) else (128,128,128)
            out_items.append(row)

        results[cat] = out_items

    return {"base_color": base_color, "results": results}

# ------------------------- API Endpoints -------------------------
@app.get("/products")
def products():
    return JSONResponse(df.to_dict(orient="records"))

events = []
def gen_event_id():
    return str(uuid.uuid4())

@app.get("/events")
def list_events():
    return JSONResponse(events)

@app.post("/events")
def add_event(title: str = Form(...), date: str = Form(...), occasion: str = Form(...)):
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except:
        return JSONResponse({"error":"date must be YYYY-MM-DD"}, status_code=400)
    ev = {"id":gen_event_id(),"title":title,"date":date,"occasion":occasion.strip().lower(),"created_at":datetime.datetime.utcnow().isoformat()}
    events.append(ev)
    return JSONResponse(ev)

@app.get("/recommend/outfit")
def recommend_outfit_for_event(event_id: str, num_outfits: int = 3):
    ev = next((e for e in events if e["id"] == event_id), None)
    if not ev:
        return JSONResponse({"error": "event not found"}, status_code=404)
    
    outfits = build_outfit(ev, num_outfits=num_outfits, per_slot_k=3)
    return JSONResponse(outfits)

@app.post("/recommend/wardrobe")
async def recommend_from_wardrobe(
    file: UploadFile = File(None),
    base_item_id: int = Form(None),
    user_prompt: str = Form(""),
    occasion: str = Form(None),
    top_k: int = Form(3)
):
    if file is None and base_item_id is None:
        return JSONResponse({"error":"upload an image file or supply base_item_id"}, status_code=400)
    contents = await file.read() if file else None
    res = wardrobe_matcher(
        wardrobe_image_bytes=contents,
        base_item_id=base_item_id,
        user_prompt=user_prompt or "",
        occasion=occasion,
        top_k_per_slot=int(top_k)
    )
    return JSONResponse(res)

@app.get("/health")
def health():
    return JSONResponse({"status":"ok","device":device,"n_products":len(df)})

# ------------------------- End -------------------------
if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
