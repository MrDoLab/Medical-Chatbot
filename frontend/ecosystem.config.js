module.exports = {
  apps: [
    {
      name: "medchat-api",
      script: "../venv/bin/uvicorn",
      args: "server:app --host 0.0.0.0 --port 8000",
      cwd: "/home/ubuntu/Medical-Chatbot/frontend",
      interpreter: "none"
    }
  ]
};

