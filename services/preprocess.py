from utils.image_utils import analyze_image
from transformers import AutoTokenizer
import numpy as np


_tokenizer = None


def get_tokenizer(name='sentence-transformers/all-MiniLM-L6-v2'):
global _tokenizer
if _tokenizer is None:
_tokenizer = AutoTokenizer.from_pretrained(name)
return _tokenizer




def preprocess_product(payload: dict) -> dict:
"""Normalize fields, compute image heuristics and token counts."""
product = payload.copy()
# Basic normalising
product['title'] = product['title'].strip()
product['price'] = float(product['price'])


# Image heuristics (inspect first image for speed)
images = payload.get('images', [])
if images:
heur = analyze_image(images[0])
else:
heur = {'resolution': (0,0), 'background_variance': 1.0, 'is_blurry': True}
product['image_quality'] = heur


# Text tokens
tokenizer = get_tokenizer()
tokens = tokenizer(product['title'], truncation=True)
product['text_tokens'] = list(tokens['input_ids'])


# Category normalisation stub
product['category_norm'] = product.get('category','_unknown_').lower()


# add metadata placeholders
product['metadata'] = {
'seller_id': product['seller_id'],
'category': product['category_norm'],
}


return product