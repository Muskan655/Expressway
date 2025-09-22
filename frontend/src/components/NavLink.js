import React from "react";
import { useNavigate } from "react-router-dom";

const NavLink = ({ text, showNewTag = false }) => {
  const navigate = useNavigate();
  const handleClick = (e) => {
    e.preventDefault();
    const result = text.replace(/\s+/g, "");
    navigate(`/${result.toLowerCase()}`);
  };

  const dest = `/${text.replace(/\s+/g, "").toLowerCase()}`;
  return (
    <a href={dest} className="nav-link" onClick={handleClick}>
      {text}
      {showNewTag && <span className="new-tag">NEW</span>}
    </a>
  );
};

export default NavLink;
