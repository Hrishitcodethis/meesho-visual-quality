from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import requests
from io import BytesIO


# Lazy-load models
_text_model = None
_clip_model = None
_clip_processor = None




def _load_text_model(name='sentence-transformers/all-MiniLM-L6-v2'):
global _text_model
if _text_model is None:
_text_model = SentenceTransformer(name)
return _text_model




def _load_clip(name='openai/clip-vit-base-patch32'):
global _clip_model, _clip_processor
if _clip_model is None:
_clip_model = CLIPModel.from_pretrained(name)
_clip_processor = CLIPProcessor.from_pretrained(name)
return _clip_model, _clip_processor




def _fetch_image(url: str):
resp = requests.get(url, timeout=5)
img = Image.open(BytesIO(resp.content)).convert('RGB')
return img




def compute_embeddings(product: dict) -> dict:
"""Return a multimodal embedding dict: {'image': vec, 'text': vec, 'combined': vec}.
This is synchronous for simplicity; production should use GPU batch workers.
"""
# Text
text_model = _load_text_model()
text_vec = text_model.encode(product['title'] + '\n' + product.get('description',''), normalize_embeddings=True)


# Image (first image)
image_vec = None
if product.get('images'):
try:
clip_model, clip_processor = _load_clip()
img = _fetch_image(product['images'][0])
inputs = clip_processor(images=img, return_tensors='pt')
with torch.no_grad():
image_out = clip_model.get_image_features(**inputs)
image_vec = image_out[0].cpu().numpy()
except Exception:
image_vec = np.zeros(512, dtype=float)
else:
image_vec = np.zeros(512, dtype=float)


# combined (simple concat + l2 norm)
combined = np.concatenate([text_vec, image_vec])
# normalise
norm = np.linalg.norm(combined) or 1.0
combined = combined / norm


return {'text': text_vec, 'image': image_vec, 'combined': combined}