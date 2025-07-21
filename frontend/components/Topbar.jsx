import { useEffect, useState } from "react";
import NewsTicker from "./NewsTicker";
import { Settings } from "lucide-react";

export default function Topbar({ setShowSettings, language = "EN", setLanguage = () => {} }) {
  const [newsList, setNewsList] = useState([
    { title: "Loading News...", link: "#" }
  ]);

  useEffect(() => {
    const apiUrl = "http://44.208.140.83:8001/api/news";

    fetch(apiUrl)
      .then(res => res.json())
      .then(data => {
        if (data.items) {
          setNewsList(data.items.map(item => ({
            title: item.title,
            link: item.link || "#"
          })));
        } else {
          setNewsList([{ title: "뉴스를 불러올 수 없습니다.", link: "#" }]);
        }
      })
      .catch(err => {
        console.error(err);
        setNewsList([{ title: "뉴스를 불러올 수 없습니다.", link: "#" }]);
      });
  }, []);

  return (
    <div className="flex flex-col w-full">
      {/* 뉴스 영역 */}
      <div className="bg-[rgba(21,101,192,0.80)] text-white text-xs px-3 py-1 font-light tracking-normal flex items-center overflow-hidden">
        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse mr-2" />
        <span className="text-gray-300 mr-3 font-bold">NEWS</span>
        <NewsTicker newsList={newsList} />
      </div>

      {/* 로고 + 타이틀 + 우측 설정 */}
      <div className="bg-gradient-to-r from-[#004080] to-[#1E3A8A] text-white flex justify-between items-center px-6 py-3 shadow border-b">
        <div className="flex items-center space-x-4">
          <img
            src="vietnam2.png"
            alt="병원 로고"
            className="h-10 w-auto p-0.5 object-contain"
          />
          <div className="flex flex-col">
            <h1 className="text-[20px] font-bold tracking-wide drop-shadow">
              Bệnh viện Đại học Nam Cần Thơ
            </h1>
            <span className="text-[11px] text-gray-300">
              Collaborative Initiative with Wonkwang University Hospital
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="border-none text-[13px] px-2 py-1 rounded-lg bg-[#1E3A8A] text-white focus:outline-none focus:ring"
          >
            <option value="EN">EN</option>
            <option value="KR">VN</option>
          </select>

          <button
            onClick={() => setShowSettings(true)}
            className="p-2 hover:bg-blue-200 rounded-full transition"
            title="Settings (Set your ID)"
          >
            <Settings className="w-5 h-5 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
}
