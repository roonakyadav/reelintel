from fastapi import FastAPI
from app.routes.analyze import router

app = FastAPI(title="ReelIntel API")

app.include_router(router)

@app.get("/")
def root():
    return {"message": "ReelIntel API is running"}