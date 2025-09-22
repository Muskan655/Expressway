import React, { useState, useEffect } from "react";
import axios from "axios";
import "./Expressway.css";

function Expressway() {
  const [events, setEvents] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [newEventTitle, setNewEventTitle] = useState("");
  const [occasion, setOccasion] = useState("");
  const [upcomingOutfits, setUpcomingOutfits] = useState([]);
  const [showWardrobe, setShowWardrobe] = useState(false);
  const [recommendations, setRecommendations] = useState({});
  const [baseImage, setBaseImage] = useState(null);
  const [baseItemId, setBaseItemId] = useState("");
  const [userPrompt, setUserPrompt] = useState("");
  const [selectedEventId, setSelectedEventId] = useState("");
  const [hoveredEventId, setHoveredEventId] = useState("");
  const [chipOpen, setChipOpen] = useState(false);
  const [showAddEvent, setShowAddEvent] = useState(false);

  // ---------------- Fetch Events ----------------
  const fetchEvents = async () => {
    try {
      const res = await axios.get("http://localhost:8000/events");
      setEvents(res.data);
      fetchUpcomingOutfits(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, []);

  // ---------------- Add Event ----------------
  const addEvent = async () => {
    if (!newEventTitle || !selectedDate || !occasion) return;

    const formData = new FormData();
    formData.append("title", newEventTitle);
    formData.append("date", selectedDate);
    formData.append("occasion", occasion);

    try {
      await axios.post("http://localhost:8000/events", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setNewEventTitle("");
      setSelectedDate("");
      setOccasion("");
      fetchEvents();
    } catch (err) {
      console.error(err.response?.data || err.message);
    }
  };

  // ---------------- Fetch Upcoming Outfits ----------------
  const fetchUpcomingOutfits = async (eventList) => {
    const today = new Date();
    const upcoming = eventList.filter((ev) => {
      if (!ev.date) return false;
      const eventDate = new Date(ev.date);
      const diffDays = Math.ceil(
        (eventDate - today) / (1000 * 60 * 60 * 24)
      );
      return diffDays <= 7 && diffDays >= 0;
    });

    const outfitsAll = [];
    for (let ev of upcoming) {
      try {
        const res = await axios.get(
          `http://localhost:8000/recommend/outfit?event_id=${ev.id}&num_outfits=3`
        );
        outfitsAll.push({ event: ev, outfits: res.data });
      } catch (err) {
        console.error(err);
      }
    }
    setUpcomingOutfits(outfitsAll);
  };

  // ---------------- Wardrobe Matcher ----------------
  const handleWardrobeSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    if (baseImage) formData.append("file", baseImage);
    if (baseItemId) formData.append("base_item_id", baseItemId);
    formData.append("user_prompt", userPrompt);
    formData.append("occasion", occasion);
    formData.append("top_k", 3);

    try {
      const res = await axios.post(
        "http://localhost:8000/recommend/wardrobe",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      if (res.data.error) alert(res.data.error);
      else setRecommendations(res.data.results);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex h-screen gap-4 font-sans bg-gradient-to-br from-purple-100 via-pink-50 to-blue-50 relative">
      {/* Top-right Wardrobe button */}
      <button
        onClick={() => setShowWardrobe(true)}
        className="btn-glass btn-medium wardrobe-top-right"
      >
        Open Wardrobe Matcher
      </button>

      {/* ---------------- Left Sidebar ---------------- */}
      <div className="w-80 bg-white/20 backdrop-blur-md border border-white/30 p-4 flex flex-col min-h-0">
        <div className="title-only mb-2">
          <div className="logo-text">Myntra Expressway</div>
        </div>
        {/* Add Event Toggle */}
        <div className="section-gap">
          <button onClick={() => setShowAddEvent((v) => !v)} className="btn-glass btn-medium add-event-btn">Add Event</button>
        </div>
        {showAddEvent && (
        <div className="mb-4 glass-card compact-card event-form-compact">
          <h2 className="font-semibold mb-2">Add Event</h2>
          <input
            type="text"
            placeholder="Event Title"
            value={newEventTitle}
            onChange={(e) => setNewEventTitle(e.target.value)}
            className="border p-2 mb-2 w-full rounded"
          />
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="border p-2 mb-2 w-full rounded"
          />
          <input
            type="text"
            placeholder="Occasion (formal, casual, party, sports)"
            value={occasion}
            onChange={(e) => setOccasion(e.target.value)}
            className="border p-2 mb-2 w-full rounded"
          />
          <button onClick={addEvent} className="btn-glass btn-compact w-auto self-start mt-2">
            Add Event
          </button>
        </div>
        )}

        {/* Events dropdown (kept), with hoverable Details button */}
        <div className="inline-fit p-2 glass-card">
          {events.length === 0 ? (
            <button type="button" className="empty-btn" disabled>No events added</button>
          ) : (
            <div className="event-select-wrap">
              <label className="text-xs mb-1 block">Select Event</label>
              <div className="event-select-row">
                <select
                  className="event-select"
                  value={selectedEventId}
                  onChange={(e) => { setSelectedEventId(e.target.value); setChipOpen(false); }}
                >
                  <option value="">Choose...</option>
                  {events.map((ev) => (
                    <option key={ev.id} value={ev.id}>{ev.title}</option>
                  ))}
                </select>
                <button
                  type="button"
                  className="btn-glass btn-compact event-info-button"
                  onMouseEnter={() => selectedEventId && setHoveredEventId(selectedEventId)}
                  onMouseLeave={() => setHoveredEventId("")}
                  disabled={!selectedEventId}
                >
                  Details
                </button>
              </div>
              {hoveredEventId && hoveredEventId === selectedEventId && (
                <div className="dropdown-popover glass-card show">
                  {(() => {
                    const ev = events.find(x => x.id === selectedEventId);
                    if (!ev) return null;
                    const today = new Date();
                    const eventDate = new Date(ev.date);
                    const diffDays = Math.ceil((eventDate - today) / (1000 * 60 * 60 * 24));
                    const upcoming = diffDays <= 7 && diffDays >= 0;
                    return (
                      <div>
                        <strong className="gradient-text">{ev.title}</strong>
                        <p className="text-xs text-gray-700">Occasion: {ev.occasion}</p>
                        <p className="text-xs text-gray-700">Date: {ev.date}</p>
                        {upcoming && (
                          <p className="text-green-600 text-xs font-semibold">Upcoming in {diffDays} day(s)!</p>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ---------------- Main Content ---------------- */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="heading-row">
          <h2 className="hero-title gradient-text title-shadow">
            Upcoming Outfits
          </h2>
        </div>
        <div className="main-chip-row">
          {(() => {
            const ev = events.find(x => x.id === selectedEventId);
            if (!ev) return null;
            return (
              <>
                <button type="button" className="event-chip" onClick={() => setChipOpen((v) => !v)}>
                  {ev.title}
                </button>
                {chipOpen && (
                  <div className="event-details-card">
                    <strong className="gradient-text">{ev.title}</strong>
                    <p className="text-xs text-gray-700">Occasion: {ev.occasion}</p>
                    <p className="text-xs text-gray-700">Date: {ev.date}</p>
                    {(() => {
                      if (!ev?.date) return null;
                      const today = new Date();
                      const eventDate = new Date(ev.date);
                      const diffDays = Math.ceil((eventDate - today) / (1000 * 60 * 60 * 24));
                      const upcoming = diffDays <= 7 && diffDays >= 0;
                      return upcoming ? (
                        <p className="text-green-600 text-xs font-semibold">Upcoming in {diffDays} day(s)!</p>
                      ) : null;
                    })()}
                  </div>
                )}
              </>
            );
          })()}
        </div>
        {upcomingOutfits.length === 0 ? (
  <p>No upcoming events within 7 days.</p>
) : (
      upcomingOutfits.map(({ event, outfits }) => (
        <div key={event.id} className="mb-4">
          {/* Event Button */}
          <div className="main-chip-row">
            <button type="button" className="event-chip" onClick={() => setChipOpen((v) => !v)}>
              {event.title}
            </button>
            {chipOpen && (
              <div className="event-details-card">
                <strong className="gradient-text">{event.title}</strong>
                <p className="text-xs text-gray-700">Occasion: {event.occasion}</p>
                <p className="text-xs text-gray-700">Date: {event.date}</p>
                {(() => {
                  if (!event?.date) return null;
                  const today = new Date();
                  const eventDate = new Date(event.date);
                  const diffDays = Math.ceil((eventDate - today) / (1000 * 60 * 60 * 24));
                  const upcoming = diffDays <= 7 && diffDays >= 0;
                  return upcoming ? (
                    <p className="text-green-600 text-xs font-semibold">Upcoming in {diffDays} day(s)!</p>
                  ) : null;
                })()}
              </div>
            )}
          </div>

          {/* Outfit Tiles */}
          {outfits.length === 0 ? (
            <p>No outfit available</p>
          ) : (
            <div className="outfits-grid">
              {outfits.map((outfit, idx) => {
                const items = [
                  { name: outfit.base?.name, image_url: outfit.base?.image_url, brand: outfit.base?.brand, price: outfit.base?.price }
                ].concat(
                  Object.values(outfit.complements || {}).filter(Boolean)
                ).slice(0, 3);
                return (
                  <div key={idx} className="outfit-tile">
                    <div className="item-trio">
                      {items.map((it, i) => (
                        <div key={i} className="item-cell">
                          <div className="image-hover-container trio-image">
                            <img src={it.image_url} alt={it.name || it.brand} className="trio-image object-cover rounded" />
                            <div className="wishlist-overlay"><button className="wishlist-btn" type="button">❤ Add to wishlist</button></div>
                          </div>
                          <p className="text-xs name-ellipsis" title={it.brand || it.name}>{it.brand || it.name}</p>
                          <p className="text-xs text-gray-700">{it.price !== undefined && it.price !== null ? `₹${it.price}` : ""}</p>
                        </div>
                      ))}
                    </div>
                    <button className="btn-glass btn-medium genai-btn" type="button">GenAI Trial</button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ))
    )}

      </div>

      {/* ---------------- Wardrobe Matcher Modal ---------------- */}
      {showWardrobe && (
      <div className="fixed inset-0 modal-backdrop flex justify-center items-start z-50">
        <div className="wardrobe-card w-96 relative">
          
          {/* Go Back button */}
          <div
            className="go-back-btn"
            onClick={() => setShowWardrobe(false)}
          >
            ← Go Back
          </div>

          <h2 className="wardrobe-title mb-4 gradient-text">Wardrobe Matcher</h2>
          <form onSubmit={handleWardrobeSubmit}>
            <input
              type="file"
              onChange={(e) => setBaseImage(e.target.files[0])}
              className="border p-2 mb-2 w-full rounded"
            />
            <input
              type="number"
              placeholder="Base Item ID"
              value={baseItemId}
              onChange={(e) => setBaseItemId(e.target.value)}
              className="border p-2 mb-2 w-full rounded"
            />
            <input
              type="text"
              placeholder="Prompt (shoes, trousers...)"
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              className="border p-2 mb-2 w-full rounded"
            />
            <input
              type="text"
              placeholder="Occasion"
              value={occasion}
              onChange={(e) => setOccasion(e.target.value)}
              className="border p-2 mb-2 w-full rounded"
            />
            <button
              type="submit"
              className="recommend-btn"
            >
              Recommend
            </button>

          </form>

          {/* Display Recommendations */}
          <div className="recommendations-wrapper">
            {recommendations &&
              Object.entries(recommendations)
                .filter(([cat, items]) => items.length > 0) // show only non-empty categories
                .map(([cat, items]) => (
                  <div key={cat} className="mb-6 mt-4">
                    <h3>{cat}</h3> {/* Bold category name */}
                    <div className="flex flex-wrap gap-4">
                      {items.map((item) => (
                        <div key={item.id} className="outfit-card">
                          <div className="image-hover-container mb-2">
                            <img
                              src={item.image_url}
                              alt={item.name}
                              className="suggestion-image object-cover rounded"
                            />
                            <div className="wishlist-overlay">
                              <button className="wishlist-btn" type="button">
                                ❤ Add to wishlist
                              </button>
                            </div>
                          </div>
                          {item.brand && (
                            <p className="text-xs text-gray-700">Brand: {item.brand}</p>
                          )}
                          {item.price !== undefined && (
                            <p className="text-xs text-gray-700">Price: ₹{item.price}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
          </div>
        </div>
      </div>
    )}

    </div>
  );
}


export default Expressway;
