import numpy as np
from db.milvus_client import search_similar


# placeholder weights; ideally tuned per category
WEIGHTS = {
'image': 0.4,
'text': 0.2,
'price': 0.2,
'performance': 0.2
}




def compute_image_similarity_score(product, emb_vector):
# For demo: use 1.0 if resolution >= 800x800 else 0.5
res = product['image_quality']['resolution']
return 1.0 if res[0] >= 800 and res[1] >= 800 else 0.5




def compute_text_score(product, neighbors):
# simple heuristic: if neighbors have similar titles
return 0.8




def compute_price_score(product, neighbors):
# compare to neighbor prices (placeholder)
return 0.7




def compute_performance_score(product, neighbors):
# placeh