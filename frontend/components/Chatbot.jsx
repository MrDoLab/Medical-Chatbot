import { useEffect, useState } from "react";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";
import ChatArea from "./ChatArea";
import InputBar from "./InputBar";
import SettingsPopup from "./SettingsPopup";

export default function ChatbotApp() {
  const [chatHistory, setChatHistory] = useState([]);
  const [inputText, setInputText] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [user, setUser] = useState({ name: "Eric Kim" });

  const [sessionList, setSessionList] = useState(() => {
    return JSON.parse(localStorage.getItem("chatSessions") || "{}");
  });
  const [activeSessionId, setActiveSessionId] = useState(null);

  const sendQuestion = async () => {
    if (!inputText.trim()) return;

    const userMessage = { role: "user", text: inputText };
    const loadingMessage = { role: "bot", isLoading: true };
    const tempHistory = [...chatHistory, userMessage, loadingMessage];

    setChatHistory(tempHistory);
    setInputText("");

    try {
      const res = await fetch("http://44.208.140.83:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: inputText, user_id: user.name }),
      });

      const data = await res.json();
      const answer = data.answer || "❌ 답변 불러오기 실패";
      const newHistory = [
        ...tempHistory.filter((m) => !m.isLoading),
        { role: "bot", text: answer },
      ];

      setChatHistory(newHistory);

      const sessionId = data.session_id;
      const sessionTitle = inputText.slice(0, 15);
      const newSessions = {
        ...sessionList,
        [sessionId]: {
          title: sessionTitle,
          messages: newHistory,
          date: new Date().toLocaleString(),
        },
      };

      setSessionList(newSessions);
      setActiveSessionId(sessionId);
      localStorage.setItem("chatSessions", JSON.stringify(newSessions));
    } catch (err) {
      console.error("❌ 에러 발생:", err);
    }
  };

  const handleSessionSelect = (sessionId) => {
    const session = sessionList[sessionId];
    if (session) {
      setChatHistory(session.messages);
      setActiveSessionId(sessionId);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <Topbar setShowSettings={setShowSettings} />
      <div className="flex flex-1">
        <Sidebar
          sessionList={sessionList}
          activeSessionId={activeSessionId}
          onSessionSelect={handleSessionSelect}
          user={user}
        />
        <div className="flex-1 flex flex-col">
          <ChatArea chatHistory={chatHistory} />
          <InputBar inputText={inputText} setInputText={setInputText} sendQuestion={sendQuestion} />
        </div>
      </div>
      <SettingsPopup open={showSettings} setOpen={setShowSettings} user={user} setUser={setUser} />
    </div>
  );
}
