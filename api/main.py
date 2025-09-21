from fastapi import FastAPI
from .routes import router


app = FastAPI(title="Meesho Visual Quality Engine")
app.include_router(router)


@app.get("/health")
async def health():
return {"status": "ok"}