import Modal from "react-modal";
export default function SettingsPopup({open, setOpen, user, setUser}) {
    const saveSetting = () => setOpen(false);
    return (
        <Modal isOpen = {open} onRequestClose ={()=> setOpen(false)} className = "bg-white p-6 rounded shadow w-96 mx-auto mt-20">
            <h2 className="text-lg font-bold mb-4">Settings</h2>
            <div className = "space-y-4">
                <div>
                    <label className="block text-sm">Username (ID):</label>
                    <input className="border rounded w-full px-2 py-1 mt-1" value={user.name} onChange={(e) => setUser({ ...user, name: e.target.value})} />
                </div>
                <div className="flex justify-end space-x-2 mt-6">
                    <button className = "border px-3 py-1" onClick={() => setOpen(false)}>close</button>
                    <button className = "border px-3 py-1 bg-blue-500 text-white" onClick={saveSetting}>save</button>
                </div>
            </div>
        </Modal>
    )
}