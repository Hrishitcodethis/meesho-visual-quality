import asyncio
from .preprocess import preprocess_product
from .embeddings import compute_embeddings
from db.qdrant_client import upsert_product
from ranking.ranker import compute_composite_score


async def process_upload(payload: dict) -> dict:
"""End-to-end mini flow: preprocess -> embeddings -> upsert -> score -> recommendations"""
product = preprocess_product(payload)
emb = compute_embeddings(product)
# Upsert to vector DB and metadata store
upsert_product(product, emb)
# Run quick similarity search + scoring
score, top_neighbors = compute_composite_score(product, emb)


if score >= 0.7:
status = "accepted"
recommendations = []
else:
status = "improve"
recommendations = generate_recommendations(product, top_neighbors)


return {
"product_id": product['product_id'],
"status": status,
"score": float(score),
"recommendations": recommendations,
"neighbors": top_neighbors[:5]
}




def generate_recommendations(product, neighbors):
# Simple rule-based suggestions — can be expanded
recs = []
if product['image_quality']['resolution'] < (800, 800):
recs.append('Increase image resolution to >= 800x800')
if product['image_quality']['background_variance'] > 0.5:
recs.append('Use clean/consistent background')
if len(product['text_tokens']) < 5:
recs.append('Add descriptive title and bullet points')
return recs