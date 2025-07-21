from fastapi import FastAPI

app = FastAPI()

@app.get("/api/list")
def list_sessions():
    return ["ok"]
