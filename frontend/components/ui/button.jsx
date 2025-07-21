export function Button({ children, onClick, variant = "solid", className = "" }) {
  const base = "px-3 py-1 rounded";
  const variants = {
    solid: "bg-blue-500 text-white",
    outline: "border border-gray-400",
  };
  return (
    <button onClick={onClick} className={`${base} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
}