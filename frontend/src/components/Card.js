import React, { useEffect, useMemo, useState } from "react";
import io from "socket.io-client";
import "./Card.css";

// socket for sharing products from catalog (connect on demand; no retries)
const shareSocket = io("/", { path: "/socket.io", autoConnect: false, reconnection: false, timeout: 1000 });

const Card = ({ images, name, description, price, originalPrice }) => {
  const [currentImage, setCurrentImage] = useState(0);
  const [hovered, setHovered] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [selectedGroupIds, setSelectedGroupIds] = useState([]);
  const [myRooms, setMyRooms] = useState([]);

  const discount = (Number(originalPrice) || 0) - (Number(price) || 0);
  const user = useMemo(() => {
    try {
      return JSON.parse(localStorage.getItem("user") || "null");
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    let interval;
    if (hovered) {
      interval = setInterval(() => {
        setCurrentImage((prev) => (prev + 1) % images.length);
      }, 1500);
    } else {
      setCurrentImage(0);
    }
    return () => clearInterval(interval);
  }, [hovered, images.length]);

  const openShare = async (e) => {
    e.stopPropagation();
    if (!user) {
      alert("Please login on Group Shopping tab first.");
      return;
    }
    const res = await fetch(`/api/rooms?username=${encodeURIComponent(user.username)}`);
    const rooms = await res.json();
    setMyRooms(rooms);
    setShowModal(true);
  };

  const toggleGroup = (roomId) => {
    setSelectedGroupIds((prev) =>
      prev.includes(roomId) ? prev.filter((id) => id !== roomId) : [...prev, roomId]
    );
  };

  const shareToSelected = async () => {
    if (!user) return;
    try { if (!shareSocket.connected) shareSocket.connect(); } catch {}
    const targets = myRooms.filter((r) => selectedGroupIds.includes(r._id));
    for (const room of targets) {
      shareSocket.emit("join_room", { roomId: room._id, username: user.username });
      await new Promise((res) => setTimeout(res, 120));
      shareSocket.emit("send_message", {
        roomId: room._id,
        sender: user.username,
        type: "product",
        content: { name, price, image: images[0] },
      });
    }
    setShowModal(false);
    setSelectedGroupIds([]);
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedGroupIds([]);
  };

  return (
    <>
      <div
        className={`card ${hovered ? "hovered" : ""}`}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <div className="image-container">
          <img src={images[currentImage]} alt={name} className="product-image" />
          <div className="rating-overlay">⭐ 4.2 | 1.6k</div>
        </div>

        <div className="card-content">
          <h3 className="product-name">{name}</h3>
          <p className="product-description">{description}</p>
          <div className="price-section">
            <span className="price">Rs. {price}</span>
            {originalPrice && <span className="original-price">Rs. {originalPrice}</span>}
            {discount > 0 && <span className="discount"> (Rs. {discount} OFF)</span>}
          </div>

          {hovered && (
            <div className="hover-info">
              <button className="wishlist-button">♡ Add to Wishlist</button>
              <button className="share-button" onClick={openShare}>
                Add to group
              </button>
            </div>
          )}
        </div>
      </div>

      {showModal && (
        <div className="modal-overlay">
          <div className="share-modal">
            <h3 className="modal-title">Select groups</h3>
            <div className="group-list">
              {myRooms.map((room) => (
                <div
                  key={room._id}
                  className={`group-list-item${selectedGroupIds.includes(room._id) ? " selected" : ""}`}
                  onClick={() => toggleGroup(room._id)}
                >
                  <span className={`checkbox${selectedGroupIds.includes(room._id) ? " checked" : ""}`} />
                  <span className="group-name">{room.name}</span>
                </div>
              ))}
              {myRooms.length === 0 && <div className="empty-groups">No groups joined yet</div>}
            </div>
            <div className="modal-actions">
              <button className="btn btn-light" onClick={closeModal}>Cancel</button>
              <button className="btn btn-primary" onClick={shareToSelected} disabled={selectedGroupIds.length === 0}>
                Share
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Card;
