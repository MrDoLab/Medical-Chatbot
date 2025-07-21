import { useEffect, useRef } from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faNotesMedical } from '@fortawesome/free-solid-svg-icons';
import ReactMarkdown from 'react-markdown';


export default function ChatArea({ chatHistory}) {
  const endRef = useRef(null);

  // 🟡 대화가 생길 때마다 아래로 자동 스크롤
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const isEmpty = chatHistory.length === 0;

  return (
    <div className="flex-1 flex flex-col px-6 py-3 shadow bg-[#f7fbff] overflow-hidden">

      {/* 상단 제목 */}
      <div className="px-6 mt-4">
        <div className="flex items-baseline mb-2 space-x-3">
          <span className="text-3xl font-bold text-[#004080]  tracking-wide">MedLink</span>
          <span className="text-[11px] text-gray-500">AI Medical Chatbot</span>
        </div>
      </div>

      {/* 채팅창 본문 */}
      <div className={`flex-1 overflow-y-auto transition-all duration-500 ${
        isEmpty ? "flex justify-center items-center" : "flex flex-col"
      } mt-6 mb-4`}>

        {/* 대화 내용 */}
        <div className="flex flex-col space-y-3 px-2 overflolw-y-auto">
        
          {chatHistory.map((chat, idx) => {
            const isUser = chat.role === 'user';
            return (
              <div
                key={idx}
                className={`flex ${isUser ? 'justify-end' : 'justify-start'} items-start gap-2`}
              >
                {!isUser && (
                  <div className="mr-2 mt-1">
                    <FontAwesomeIcon icon={faNotesMedical} className="h-7 w-7 text-green-600" />
                  </div>
                )}
                <div
                  className={`max-w-[70%] p-3 py-2 rounded-xl text-sm whitespace-pre-wrap ${
                    isUser
                      ? 'text-gray-800 bg-[#edf4ff] border border-[#98cfff] shadow ml-auto'
                      : 'text-gray-800 bg-[#e6f8df] border border-green-300 shadow'
                  }`}
                >
                  {chat.isLoading ? (
                    <div className="flex items-center space-x-2 animate-pulse">
                      <svg className="animate-spin h-4 w-4 text-green-500" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 100 16v-4l-3 3 3 3v-4a8 8 0 01-8-8z" />
                      </svg>
                      <span className="text-sm text-gray-500">Thinking...</span>
                    </div>
                  ) : (
                    <ReactMarkdown>{chat.text}</ReactMarkdown>
                  )}
                </div>
              </div>
            );
          })}
          {/* 자동 스크롤용 div */}
          <div ref={endRef}></div>
        </div>

        {/* 처음이면 안내 메시지 */}
        {isEmpty && (
          <div className="absolute top-1/2 text-gray-500 text-sm mx-auto text-center">
            " How can I assist you today? "
          </div>
        )}
      </div>

      {/* 버튼 영역은 그대로 유지 */}
      <div className="pt-3 flex space-x-2 flex-wrap mt-3">
        {/* QuickButton 등 생략 */}
      </div>
    </div>
  );
}
