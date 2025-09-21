from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import os
import numpy as np

MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
COLLECTION_NAME = os.getenv('MILVUS_COLLECTION', 'products')

# Track connection and collection
_collection = None


def client():
    """
    Connect to Milvus and ensure collection exists.
    Returns a Collection object.
    """
    global _collection
    if _collection is not None:
        return _collection

    # Connect to Milvus
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Define schema if collection does not exist
    if COLLECTION_NAME not in [c.name for c in Collection.list_collections()]:
        fields = [
            FieldSchema(name="product_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="seller_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        schema = CollectionSchema(fields=fields, description="Product embeddings")
        _collection = Collection(name=COLLECTION_NAME, schema=schema)
    else:
        _collection = Collection(name=COLLECTION_NAME)

    return _collection


def upsert_product(product: dict, embeddings: dict):
    """
    Insert or update a product with its embedding in Milvus.
    """
    c = client()
    vector = embeddings['combined'].tolist()

    data = [
        [product['product_id']],     # product_id
        [product['title']],          # title
        [product['seller_id']],      # seller_id
        [float(product['price'])],   # price
        [product['category_norm']],  # category
        [vector],                    # embedding
    ]

    c.insert(data)
    c.flush()


def search_similar(vector: np.ndarray, top_k: int = 10, category: str = None):
    """
    Search for similar products in Milvus.
    """
    c = client()
    c.load()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = c.search(
        data=[vector.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["product_id", "title", "seller_id", "price", "category"],
    )

    return results
