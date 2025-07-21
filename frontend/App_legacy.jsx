import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";

export default function App() {
  const [chatHistory, setChatHistory] = useState(() => {
    const saved = localStorage.getItem("chatHistory");
    return saved ? JSON.parse(saved) : [];
  });

  const [user, setUser] = useState({ name: "Eric" });

  const handleSessionSelect = async (sessionId) => {
    try {
      const res = await fetch(`http://44.208.140.83:8000/api/load?session_id=${sessionId}`);
      const data = await res.json();
      setChatHistory(data);
      localStorage.setItem("chatHistory", JSON.stringify(data));  // ✅ 로컬 저장
    } catch (err) {
      console.error("세션 불러오기 실패:", err);
    }
  };

  return (
    <div className="flex h-screen">
      <Sidebar user={user} onSessionSelect={handleSessionSelect} />
      <ChatArea chatHistory={chatHistory} />
    </div>
  );
}
