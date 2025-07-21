import React from "react";

export default function Sidebar({ sessionList, activeSessionId, onSessionSelect, user }) {
  return (
    <div className="bg-gradient-to-b from-[#e6f0ff] to-[#d3e3f9] text-gray-800 w-64 border-r h-full p-4 flex flex-col">
      <div className="mb-6">
        <h2 className="font-bold text-xl text-[#1E3A8A]">Medical Assistant</h2>
      </div>

      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-600">Chat History</h3>
        {Object.entries(sessionList).map(([sessionId, { title, date }]) => (
          <div
            key={sessionId}
            onClick={() => onSessionSelect(sessionId)}
            className={`flex flex-col border rounded p-2 cursor-pointer hover:bg-blue-100 ${
              activeSessionId === sessionId ? "bg-blue-200" : ""
            }`}
          >
            <span className="font-medium text-sm truncate">{title}</span>
            <span className="text-[10px] text-gray-500">{date}</span>
          </div>
        ))}
      </div>

      <div className="mt-auto flex items-center space-x-3 px-4 py-3 text-gray-700 text-sm">
        <span className="text-2xl bg-white shadow border p-1 rounded-full">🩺</span>
        <div className="flex flex-col leading-tight text-xs">
          <span className="font-semibold text-sm">Welcome Back</span>
          <span className="text-xs">ID @ {user?.name || "Guest"}</span>
        </div>
      </div>
    </div>
  );
}
