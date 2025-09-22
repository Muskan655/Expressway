import React, { useEffect, useMemo, useRef, useState } from "react";
import io from "socket.io-client";
import "./GroupChat.css";

const socket = io("/", { path: "/socket.io", autoConnect: false, reconnection: false, timeout: 1000 });

const GroupChat = () => {

  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [user, setUser] = useState(null); 

  // data
  const [rooms, setRooms] = useState([]);
  const [pendingRooms, setPendingRooms] = useState([]);
  const [currentRoom, setCurrentRoom] = useState(null);
  const [messages, setMessages] = useState([]);
  const [onlineUsers, setOnlineUsers] = useState([]);
  const [text, setText] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [allUsers, setAllUsers] = useState([]);
  const [invited, setInvited] = useState([]);
  const [newRoomName, setNewRoomName] = useState("");
  const [usersLoading, setUsersLoading] = useState(false);

  const messagesEndRef = useRef(null);
  const me = useMemo(() => (user ? user.username : ""), [user]);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("user") || "null");
      if (saved) {
        setUser(saved);
        refreshRooms(saved.username);
      }
    } catch {}
    const onAuth = (e) => {
      const u = e?.detail?.user || null;
      setUser(u);
      if (!u) {
        setRooms([]);
        setPendingRooms([]);
        setCurrentRoom(null);
        setMessages([]);
      } else {
        refreshRooms(u.username);
      }
    };
    window.addEventListener("auth-changed", onAuth);
    return () => window.removeEventListener("auth-changed", onAuth);
  }, []);

  // socket listeners
  useEffect(() => {
    if (!currentRoom) return () => {};

    const onMessage = (msg) => setMessages((prev) => [...prev, msg]);

    const onSystem = (m) => {
      setMessages((prev) => [
        ...prev,
        {
          _id: `sys-${Date.now()}`,
          type: "system",
          content: { text: m.text },
          sender: "system",
          votes: 0,
          voters: [],
          roomId: currentRoom ? currentRoom._id : undefined,
        },
      ]);
    };

    const onVoteUpdated = ({ messageId, votes }) => {
      setMessages((prev) => prev.map((m) => (m._id === messageId ? { ...m, votes } : m)));
    };

    const handleUserOnline = ({ username }) => {
      setOnlineUsers((prev) => [...new Set([...prev, username])]);
    };

    const handleUserOffline = ({ username }) => {
      setOnlineUsers((prev) => prev.filter((u) => u !== username));
    };

    try {
      if (!socket.connected) socket.connect();
    } catch {}

    socket.on("message", onMessage);
    socket.on("system_message", onSystem);
    socket.on("vote_updated", onVoteUpdated);
    socket.on("user_online", handleUserOnline);
    socket.on("user_offline", handleUserOffline);

    return () => {
      socket.off("message", onMessage);
      socket.off("system_message", onSystem);
      socket.off("vote_updated", onVoteUpdated);
      socket.off("user_online", handleUserOnline);
      socket.off("user_offline", handleUserOffline);
      try {
        socket.disconnect();
      } catch {}
    };
  }, [currentRoom]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // API 
    const api = async (path, opts = {}) => {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  };


  // auth
  const handleSignup = async () => {
    const body = { username, email, password, displayName };
    await api("/api/signup", { method: "POST", body: JSON.stringify(body) });
    const data = await api("/api/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    setUser(data.user);
    localStorage.setItem("user", JSON.stringify(data.user));
    await refreshRooms(data.user.username);
  };

  const handleLogin = async () => {
    const data = await api("/api/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    setUser(data.user);
    localStorage.setItem("user", JSON.stringify(data.user));
    await refreshRooms(data.user.username);
  };

  // rooms
  const refreshRooms = async (uname = me) => {
    if (!uname) return;
    const [joined, pending] = await Promise.all([
      api(`/api/rooms?username=${encodeURIComponent(uname)}`),
      api(`/api/rooms/pending?username=${encodeURIComponent(uname)}`),
    ]);
    setRooms(joined);
    setPendingRooms(pending);
  };

  const handleCreateRoom = async () => {
    setShowCreate(true);
    setNewRoomName("");
    setInvited([]);
    setUsersLoading(true);
    try {
      const users = await api("/api/users");
      const filtered = users.filter((u) => u.username?.toLowerCase() !== me?.toLowerCase());
      setAllUsers(filtered);
    } finally {
      setUsersLoading(false);
    }
  };

  const toggleInvite = (uname) => {
    setInvited((prev) => (prev.includes(uname) ? prev.filter((u) => u !== uname) : [...prev, uname]));
  };

  const confirmCreateRoom = async () => {
    if (!newRoomName.trim()) return;
    const room = await api("/api/rooms", {
      method: "POST",
      body: JSON.stringify({ name: newRoomName.trim(), creator: me, invited }),
    });
    setRooms((prev) => [...prev, room]);
    setShowCreate(false);
  };

  const handleAcceptInvite = async (roomId) => {
    await api(`/api/rooms/${roomId}/join`, {
      method: "POST",
      body: JSON.stringify({ username: me }),
    });
    await refreshRooms();
  };

  const handleJoinRoom = async (room) => {
    setCurrentRoom(room);
    setMessages([]);

    const history = await api(`/api/rooms/${room._id}/messages`);
    setMessages(history);

    try {
      if (!socket.connected) socket.connect();
    } catch {}
    socket.emit("join_room", { roomId: room._id, username: me });
  };



  const handleSendText = async () => {
    if (!currentRoom || !text.trim()) return;
    try {
      if (!socket.connected) socket.connect();
    } catch {}
    socket.emit("send_message", {
      roomId: currentRoom._id,
      sender: me,
      type: "text",
      content: text.trim(),
    });
    setText("");
  };

  const handleVote = async (messageId, delta) => {
    await api(`/api/messages/${messageId}/vote`, {
      method: "POST",
      body: JSON.stringify({ username: me, vote: delta }),
    });
  };

  if (!user) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        fontSize: "3rem",
        fontWeight: "bold",
        textAlign: "center",
      }}
    >
      Login or Create Account
    </div>
  );
}

    
  return (
    <div className="group-chat-container">
      <div className="sidebar">
        <div className="sidebar-greeting">Hi, {user.username}</div>
        <button onClick={handleCreateRoom} className="create-room-btn">+ Create Room</button>

        <div className="section-title">Online Users</div>
        <div className="online-users-list">
          {onlineUsers.length === 0 && <div className="muted">No one online</div>}
          {onlineUsers.map((u) => (
            <div key={u} className="online-user">🟢 {u}</div>
          ))}
        </div>

        <div className="section-title">Pending Invites</div>
        {pendingRooms.length === 0 && <div className="muted">No invites</div>}
        {pendingRooms.map((r) => (
          <div key={r._id} className="group-item" onClick={() => handleAcceptInvite(r._id)}>
            {r.name} — Tap to join
          </div>
        ))}

        <div className="section-title with-top">Your Rooms</div>
        {rooms.map((r) => (
          <div key={r._id} className="group-item room-row" onClick={() => handleJoinRoom(r)}>
            <div className="room-name">{r.name}</div>
          </div>
        ))}
      </div>

      <div className="main-content">
        {!currentRoom ? (
          <div className="chat-panel centered-empty">
            <div className="muted">Select a room to start chatting</div>
          </div>
        ) : (
          <div className="chat-panel">
            <div className="chat-messages">
              {messages.map((m) => {
                if (m.type === "system") {
                  return (
                    <div key={m._id} className="chat-msg other system-msg">
                      {m.content?.text}
                    </div>
                  );
                }
                const mine = m.sender?.toLowerCase() === me.toLowerCase();
                return (
                  <div key={m._id} className={`chat-msg ${mine ? "me" : "other"}`}>
                    {m.type === "text" && (
                      <div className="message">
                        <span className="sender">{m.sender}:</span>
                        <span className="content">{m.content}</span>
                      </div>
                    )}
                      {m.type === "product" && (
                      <div className="product-msg">
                        <div className="product-row">
                          {m.content?.image && <img src={m.content.image} alt={m.content?.name || "product"} />}
                          <div>
                            <div className="product-name-row">{m.content?.name}</div>
                            {m.content?.price != null && <div>Rs. {m.content.price}</div>}
                            <div className="vote-row">
                              <button onClick={() => handleVote(m._id, 1)}>👍</button>
                              <button onClick={() => handleVote(m._id, -1)}>👎</button>
                              <span className="score-chip">{m.votes ?? 0}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </div>

            <div className="composer">
              <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type a message"
                className="composer-input"
              />
              <button onClick={handleSendText}>Send</button>
            </div>
          </div>
        )}
      </div>

      {showCreate && (
        <div className="modal-overlay">
          <div className="room-modal">
            <h3 className="modal-title">Create new room</h3>
            <div className="form-field"><input placeholder="Room name" value={newRoomName} onChange={(e) => setNewRoomName(e.target.value)} /></div>
            <div className="user-list">
              {usersLoading && <div className="muted">Loading users…</div>}
              {!usersLoading && allUsers.map((u) => (
                <div key={u.username} className={`user-list-item${invited.includes(u.username) ? " selected" : ""}`} onClick={() => toggleInvite(u.username)}>
                  <span className={`checkbox${invited.includes(u.username) ? " checked" : ""}`} />
                  <div className="user-info">
                    <div className="user-name">{u.username}</div>
                    {u.displayName && <div className="user-display muted">{u.displayName}</div>}
                  </div>
                </div>
              ))}
            </div>
            <div className="modal-actions">
              <button className="btn btn-light" onClick={() => setShowCreate(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={confirmCreateRoom} disabled={!newRoomName.trim()}>Create</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GroupChat;
