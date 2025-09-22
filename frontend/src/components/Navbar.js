import React, { useEffect, useMemo, useRef, useState } from "react";
import NavLink from "./NavLink";
import "../App.css";
import "./Navbar.css";
import myntraLogo from '../assets/Myntra-logo.png';

import { useNavigate } from "react-router-dom";
function Navbar() {
  const navigate = useNavigate();
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem("user") || "null"); } catch { return null; }
  });
  const [open, setOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const onAuthChanged = (e) => {
      setUser(e?.detail?.user || null);
      setOpen(false);
    };
    window.addEventListener("auth-changed", onAuthChanged);
    const onClick = (e) => { if (menuRef.current && !menuRef.current.contains(e.target)) setOpen(false); };
    document.addEventListener("click", onClick);
    return () => {
      window.removeEventListener("auth-changed", onAuthChanged);
      document.removeEventListener("click", onClick);
    };
  }, []);

  const openAuth = (mode) => {
    try { window.dispatchEvent(new CustomEvent("open-auth", { detail: { mode } })); } catch {}
    const ts = Date.now();
    navigate(`/?auth=${mode}&ts=${ts}`);
  };

  const handleLogout = () => {
    try { localStorage.removeItem("user"); } catch {}
    try { window.dispatchEvent(new CustomEvent("auth-changed", { detail: { user: null } })); } catch {}
    navigate("/");
  };

  const initial = useMemo(() => (user?.displayName || user?.username || "").trim().charAt(0).toUpperCase(), [user]);

  return (
    <header className="header">
      <div className="header-content">
        {/* Logo */}
        <div className="logo">
          <div className="myntra-logo">
            <a href="/">
              <img src={myntraLogo} alt="Myntra Logo" width={80} height={60} />
            </a>
          </div>
        </div>

        {/* Navigation */}
        <nav className="navigation">
          <NavLink text="MEN" />
          <NavLink text="WOMEN" />
          <NavLink text="KIDS" />
          <NavLink text="HOME" />
          <NavLink text="BEAUTY" />
          <NavLink text="GENZ" />
          <NavLink text="EXPRESSWAY" />
          <NavLink text="GROUP SHOPPING" />
        </nav>

        {/* Search Bar */}
        <div className="search-container">
          <div className="search-bar">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              placeholder="Search for products, brands and more"
            />
          </div>
        </div>

        {/* Profile Menu */}
        <div className="profile-container" ref={menuRef}>
          <button type="button" className="profile-button" onClick={() => setOpen((v) => !v)} aria-haspopup="menu" aria-expanded={open}>
            {user ? initial || "👤" : "👤"}
          </button>
          {open && (
            <div className="profile-dropdown" role="menu">
              {!user && (
                <>
                  <button type="button" className="dropdown-item login-item" onClick={() => openAuth("login")}>Login</button>
                  <button type="button" className="dropdown-item signup-item" onClick={() => openAuth("signup")}>Sign up</button>
                </>
              )}
              {user && (
                <>
                  <div className="dropdown-label"><strong>Signed in as</strong> {user.displayName || user.username}</div>
                  <button type="button" className="dropdown-item logout-item" onClick={handleLogout}>Logout</button>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Navbar;
