import Marquee from "react-fast-marquee";
export default function NewsTicker({newsList}) {
    return (
        <div className = "w-full text-[12px] overflow-hidden">
            <Marquee speed = {30} pauseOnHover gradient={false} className="overflow-hidden whitespace-nowrap no-scrollbar">
                {newsList.map((item, idx) => (
                <a
                    key={idx}
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="mx-4 hover:underline hover:text-blue-200 transition"
                
                >
                    {item.title}
                </a>
                ))}
            </Marquee>
        </div>
    );
}
