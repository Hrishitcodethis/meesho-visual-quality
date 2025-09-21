from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.ingest import process_upload


router = APIRouter()


class UploadPayload(BaseModel):
seller_id: int
product_id: str
title: str
description: str = ""
price: float
images: list[str]
category: str


@router.post('/upload')
async def upload(payload: UploadPayload):
# Lightweight validation
try:
result = await process_upload(payload.dict())
return result
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))