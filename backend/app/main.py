from fastapi import FastAPI

app = FastAPI(title="ReelIntel API")

@app.get("/")
def root():
    return {"message": "ReelIntel API is running"}