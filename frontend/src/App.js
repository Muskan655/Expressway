import React from "react";
import "./App.css";
import Navbar from "./components/Navbar";
import HomePage from "./components/HomePage";
import MenPage from "./components/MenPage";
import { Routes, Route } from "react-router-dom";
import WomenPage from "./components/WomenPage";
import KidsPage from "./components/KidsPage";
import GroupChat from "./components/GroupChat";
import Expressway from "./components/Expressway";
function App() {
  return (
    <div className="App">
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/men" element={<MenPage />} />
        <Route path="/women" element={<WomenPage />} />
        <Route path="/kids" element={<KidsPage />} />
        <Route path="/groupshopping" element={<GroupChat />} />
        <Route path="/expressway" element={<Expressway />} />
      </Routes>
    </div>
  );
}

export default App;
