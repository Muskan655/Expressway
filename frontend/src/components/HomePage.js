import "./HomePage.css";
import myntraLogo from "../assets/Myntra-logo.png";
import banner from "../assets/banner.png";
import React, { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
function HomePage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const location = useLocation();
  const navigate = useNavigate();
  const me = useMemo(() => {
    try { return JSON.parse(localStorage.getItem("user") || "null"); } catch { return null; }
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const mode = params.get("auth");
    if (mode === "login" || mode === "signup") {
      setAuthMode(mode);
      setShowAuth(true);
    } else {
      setShowAuth(false);
    }
  }, [location.search]);

  useEffect(() => {
    const handler = (e) => {
      const mode = e?.detail?.mode || "login";
      setAuthMode(mode);
      setShowAuth(true);
    };
    window.addEventListener("open-auth", handler);
    return () => window.removeEventListener("open-auth", handler);
  }, []);

  const closeAuth = () => {
    setShowAuth(false);
    const params = new URLSearchParams(location.search);
    params.delete("auth");
    navigate({ pathname: "/", search: params.toString() });
  };

  const api = async (path, opts = {}) => {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  };

  const handleSignup = async () => {
    const body = { username, email, password, displayName };
    await api("/api/signup", { method: "POST", body: JSON.stringify(body) });
    const data = await api("/api/login", { method: "POST", body: JSON.stringify({ email, password }) });
    localStorage.setItem("user", JSON.stringify(data.user));
    try { window.dispatchEvent(new CustomEvent("auth-changed", { detail: { user: data.user } })); } catch {}
    closeAuth();
    //navigate("/groupshopping");
  };

  const handleLogin = async () => {
    const data = await api("/api/login", { method: "POST", body: JSON.stringify({ email, password }) });
    localStorage.setItem("user", JSON.stringify(data.user));
    try { window.dispatchEvent(new CustomEvent("auth-changed", { detail: { user: data.user } })); } catch {}
    closeAuth();
   // navigate("/groupshopping");
  };

  return (
    <>
      {/* Promotional Banners */}
      <div className="promo-banners">
        <div className="promo-banner left">
          <span className="promo-text">FLAT ₹400 OFF</span>
        </div>
        <div className="promo-banner right">
          <span className="promo-text">
            On Your 1st Purchase Via Myntra App!
          </span>
          <div className="myntra-logo-small">
            <img src={myntraLogo} alt="Myntra Logo" width={50} height={30} />
          </div>
        </div>
      </div>

      {/* Main Promotional Banner */}
      <div className="main-banner">
        <img src={banner} alt="Main Banner" width={1400} height={600} />
      </div>

      {/* Sponsor Section */}
      <div className="sponsor-section">
        <div className="sponsor-item">
          <div className="sponsor-label">ASSOCIATE SPONSOR</div>
          <div className="sponsor-logo">🏇 U.S. POLO ASSN. SINCE 1890</div>
          <span className="sponsor-arrow">→</span>
        </div>
        <div className="sponsor-item">
          <div className="sponsor-label">TITLE SPONSOR</div>
          <div className="sponsor-logo">🐆 PUMA</div>
          <span className="sponsor-arrow">→</span>
        </div>
        <div className="sponsor-item">
          <div className="sponsor-label">ASSOCIATE SPONSOR</div>
          <div className="sponsor-logo">JACK&JONES</div>
          <span className="sponsor-arrow">→</span>
        </div>
      </div>

      {/* Cashback Section */}
      <div className="cashback-section">
        <div className="cashback-content">
          <div className="bank-logos">
            <span className="bank-logo">🛒 Flipkart</span>
            <span className="bank-logo">🏦 AXIS BANK</span>
          </div>
          <div className="cashback-text">Flat 7.5% Cashback*</div>
        </div>
      </div>

      {/* Auth Modal */}
      {showAuth && !me && (
        <div className="auth-modal-overlay" onClick={closeAuth}>
          <div className="auth-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="auth-tabs">
              <button className={`auth-tab ${authMode === "login" ? "active" : ""}`} onClick={() => setAuthMode("login")}>Login</button>
              <button className={`auth-tab ${authMode === "signup" ? "active" : ""}`} onClick={() => setAuthMode("signup")}>Sign up</button>
            </div>
            {authMode === "login" ? (
              <div className="auth-fields">
                <input placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
                <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
                <div className="auth-actions">
                  <button className="btn btn-light" onClick={closeAuth}>Cancel</button>
                  <button className="btn btn-primary" onClick={handleLogin}>Login</button>
                </div>
              </div>
            ) : (
              <div className="auth-fields">
                <input placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} />
                <input placeholder="Display name (optional)" value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
                <input placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
                <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
                <div className="auth-actions">
                  <button className="btn btn-light" onClick={closeAuth}>Cancel</button>
                  <button className="btn btn-accent" onClick={handleSignup}>Create account</button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Floating Notification */}
      <div className="floating-notification">🔔</div>
    </>
  );
}

export default HomePage;
