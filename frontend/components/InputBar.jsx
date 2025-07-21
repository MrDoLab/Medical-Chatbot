import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowUp } from '@fortawesome/free-solid-svg-icons';
import { useRef, useEffect } from "react";

export default function InputBar({ inputText, setInputText, sendQuestion }) {
  const textareaRef = useRef(null);

  // ✅ 입력 내용에 따라 높이 자동 조절
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [inputText]);

  return (
    <div className="p-6 flex items-center justify-center bg-[#f7fbff]">
      <div className="flex items-center w-full max-w-[700px]">
        <textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendQuestion();
            }
          }}
          placeholder="Type your question and Press Enter : (e.g. symptoms, clinical guidance)"
          rows={1}
          className="ml-20 mr-4 resize-none text-sm flex-grow border-[#c6dbf7] bg-[#e6f0ff] rounded-3xl px-6 py-3.5 outline-none shadow-md ring-1 ring-blue-200 focus:ring-2 focus:ring-blue-400 leading-relaxed"
        />
        <button
          onClick={sendQuestion}
          className="ml-4 p-1 rounded-full bg-[rgba(21,101,192,0.80)] hover:bg-[#004080] hover:scale-110 shadow-xl text-white transition"
        >
          <FontAwesomeIcon icon={faArrowUp} className="w-6 h-4" />
        </button>
      </div>
    </div>
  );
}
