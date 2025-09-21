import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn.functional as F


# 1. Simple transforms for images (Note: CLIP handles its own preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# 2. CLIP visual encoder
visual_encoder = SentenceTransformer("clip-ViT-B-32")


def get_image_embedding(img_path: str) -> np.ndarray:
    """
    Encode an image into an embedding vector.
    """
    image = Image.open(img_path).convert("RGB")
    emb = visual_encoder.encode(image, convert_to_numpy=True)
    return emb


def build_embedding_matrix(dataset_path: str, max_images: int = None):
    """
    Create embeddings for all images in dataset.
    
    Args:
        dataset_path: Path to dataset folder
        max_images: Maximum number of images to process (None for all)
    """
    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(dataset_path, ext)))
    
    if not img_paths:
        raise ValueError(f"No images found in {dataset_path}")
    
    # Limit to first N images if specified
    if max_images is not None:
        img_paths = img_paths[:max_images]
        print(f"Limiting processing to first {len(img_paths)} images (max_images={max_images})")
    
    embeddings = []
    products = []

    for idx, path in enumerate(img_paths):
        try:
            emb = get_image_embedding(path)
            embeddings.append(emb)
            products.append({
                "product_id": f"mock_{idx}",
                "title": f"Product {idx}",
                "path": path
            })
            print(f"Encoded {idx+1}/{len(img_paths)}: {path} -> shape {emb.shape}")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    print(f"\nSuccessfully processed {len(embeddings)} images")
    return np.array(embeddings), products


def compute_similarity(embeddings: np.ndarray, products: list, query_idx: int = 0, top_k: int = 5, method: str = "cosine", export_to_csv: bool = False, csv_filename: str = None):
    """
    Compute semantic similarity for a query image vs dataset.
    
    Args:
        embeddings: Input embeddings array
        products: List of product dictionaries
        query_idx: Index of query embedding
        top_k: Number of top results to return
        method: Similarity method ('cosine', 'dot_product', 'euclidean', 'manhattan')
        export_to_csv: Whether to export results to CSV
        csv_filename: Custom CSV filename (optional)
    """
    query_emb = embeddings[query_idx].reshape(1, -1)
    
    if method == "cosine":
        # Cosine similarity - RECOMMENDED for CLIP embeddings
        query_tensor = torch.tensor(query_emb, dtype=torch.float32)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        sims = F.cosine_similarity(query_tensor, embeddings_tensor, dim=1).numpy()
        
    elif method == "dot_product":
        # Dot product similarity (only if embeddings are normalized)
        # CLIP embeddings from sentence-transformers are typically normalized
        sims = np.dot(query_emb, embeddings.T)[0]
        
    elif method == "euclidean":
        # Euclidean distance (converted to similarity)
        distances = euclidean_distances(query_emb, embeddings)[0]
        # Convert distance to similarity (smaller distance = higher similarity)
        sims = 1 / (1 + distances)
        
    elif method == "manhattan":
        # Manhattan distance (L1 norm)
        distances = np.sum(np.abs(query_emb - embeddings), axis=1)
        sims = 1 / (1 + distances)
        
    elif method == "semantic_weighted":
        # Custom semantic similarity with learned weights
        weights = np.ones(embeddings.shape[1])  # Replace with learned weights
        weighted_query = query_emb * weights
        weighted_embeddings = embeddings * weights
        # Use cosine similarity on weighted embeddings
        weighted_query_tensor = torch.tensor(weighted_query, dtype=torch.float32)
        weighted_embeddings_tensor = torch.tensor(weighted_embeddings, dtype=torch.float32)
        sims = F.cosine_similarity(weighted_query_tensor, weighted_embeddings_tensor, dim=1).numpy()
        
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    # Create comprehensive results with all scores
    all_results = []
    for idx, (prod, score) in enumerate(zip(products, sims)):
        all_results.append({
            'rank': idx + 1,
            'product_id': prod['product_id'],
            'title': prod['title'],
            'path': prod['path'],
            'similarity_score': score,
            'is_query': idx == query_idx,
            'query_product_id': products[query_idx]['product_id'],
            'query_path': products[query_idx]['path'],
            'similarity_method': method
        })
    
    # Sort by similarity score
    all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Update ranks after sorting
    for i, result in enumerate(all_results):
        result['rank'] = i + 1
    
    # Get top-k results
    top_results = all_results[:top_k]

    print(f"\n🔎 Semantic Similarity Results ({method}):")
    for result in top_results:
        print(f"Rank {result['rank']}: Product ID: {result['product_id']} | Path: {result['path']} | Score: {result['similarity_score']:.4f}")
    
    # Export to CSV if requested
    if export_to_csv:
        if csv_filename is None:
            query_id = products[query_idx]['product_id']
            csv_filename = f"similarity_results_{query_id}_{method}.csv"
        
        export_similarity_to_csv(all_results, csv_filename, method, products[query_idx])
    
    return top_results, all_results


def export_similarity_to_csv(results: list, filename: str, method: str, query_product: dict):
    """
    Export similarity results to CSV file.
    
    Args:
        results: List of result dictionaries
        filename: Output CSV filename
        method: Similarity method used
        query_product: Query product information
    """
    df = pd.DataFrame(results)
    
    # Add metadata as comments or separate columns
    metadata_df = pd.DataFrame([{
        'metadata_type': 'query_info',
        'query_product_id': query_product['product_id'],
        'query_title': query_product['title'],
        'query_path': query_product['path'],
        'similarity_method': method,
        'total_comparisons': len(results),
        'timestamp': pd.Timestamp.now().isoformat()
    }])
    
    # Save main results
    df.to_csv(filename, index=False)
    
    # Save metadata to separate file
    metadata_filename = filename.replace('.csv', '_metadata.csv')
    metadata_df.to_csv(metadata_filename, index=False)
    
    print(f"✅ Exported {len(results)} similarity results to: {filename}")
    print(f"✅ Exported metadata to: {metadata_filename}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"  - Method: {method}")
    print(f"  - Query: {query_product['product_id']}")
    print(f"  - Total comparisons: {len(results)}")
    print(f"  - Highest score: {max(result['similarity_score'] for result in results):.4f}")
    print(f"  - Lowest score: {min(result['similarity_score'] for result in results):.4f}")
    print(f"  - Average score: {np.mean([result['similarity_score'] for result in results]):.4f}")


def export_comparison_matrix(embeddings: np.ndarray, products: list, method: str = "cosine", filename: str = "similarity_matrix.csv"):
    """
    Export a full similarity matrix comparing ALL images to ALL images.
    Warning: This can be large! N×N matrix where N is number of images.
    
    Args:
        embeddings: Input embeddings array
        products: List of product dictionaries
        method: Similarity method
        filename: Output CSV filename
    """
    n_products = len(products)
    print(f"🔄 Computing full {n_products}×{n_products} similarity matrix...")
    
    # Compute full similarity matrix
    if method == "cosine":
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    elif method == "dot_product":
        similarity_matrix = np.dot(embeddings, embeddings.T)
    elif method == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings)
        similarity_matrix = 1 / (1 + distances)
    else:
        raise ValueError(f"Method {method} not supported for full matrix")
    
    # Create DataFrame with product IDs as row/column names
    product_ids = [prod['product_id'] for prod in products]
    df = pd.DataFrame(similarity_matrix, index=product_ids, columns=product_ids)
    
    # Export to CSV
    df.to_csv(filename)
    print(f"✅ Exported {n_products}×{n_products} similarity matrix to: {filename}")
    
    return df


def batch_export_similarities(embeddings: np.ndarray, products: list, query_indices: list = None, method: str = "cosine", output_dir: str = "similarity_results"):
    """
    Export similarity results for multiple query images.
    
    Args:
        embeddings: Input embeddings array
        products: List of product dictionaries
        query_indices: List of query indices (if None, uses all images as queries)
        method: Similarity method
        output_dir: Output directory for CSV files
    """
    if query_indices is None:
        query_indices = list(range(len(products)))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Batch processing {len(query_indices)} queries...")
    
    all_batch_results = []
    
    for i, query_idx in enumerate(query_indices):
        print(f"Processing query {i+1}/{len(query_indices)}: {products[query_idx]['product_id']}")
        
        # Get filename for this query
        query_id = products[query_idx]['product_id']
        csv_filename = os.path.join(output_dir, f"similarity_{query_id}_{method}.csv")
        
        # Compute similarities
        top_results, all_results = compute_similarity(
            embeddings, products, query_idx=query_idx, 
            top_k=len(products), method=method, 
            export_to_csv=True, csv_filename=csv_filename
        )
        
        # Add to batch results
        for result in all_results:
            result['batch_query_idx'] = query_idx
            all_batch_results.append(result)
    
    # Export combined batch results
    batch_filename = os.path.join(output_dir, f"batch_similarity_results_{method}.csv")
    batch_df = pd.DataFrame(all_batch_results)
    batch_df.to_csv(batch_filename, index=False)
    
    print(f"✅ Batch export complete! Results saved to: {output_dir}")
    print(f"✅ Combined results saved to: {batch_filename}")
    
    return all_batch_results
    """
    CLIP-optimized similarity computation using the model's native approach.
    """
    # CLIP embeddings are typically L2 normalized, so dot product = cosine similarity
    query_emb = embeddings[query_idx]
    
    # Ensure embeddings are normalized (CLIP usually does this automatically)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_query = query_emb / np.linalg.norm(query_emb)
    
    # Compute similarities (dot product on normalized vectors = cosine similarity)
    sims = np.dot(normalized_query, normalized_embeddings.T)
    
    ranked = sorted(
        zip(products, sims),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    print("\n🔎 CLIP-Optimized Similarity Results:")
    for prod, score in ranked:
        print(f"Product ID: {prod['product_id']} | Path: {prod['path']} | Score: {score:.4f}")
    
    return ranked


def compute_learned_semantic_similarity(embeddings: np.ndarray, products: list, query_idx: int = 0, top_k: int = 5):
    """
    Advanced semantic similarity using learned transformations.
    """
    # Example: Apply a learned transformation matrix
    # For CLIP, you might learn domain-specific projections
    transformation_matrix = np.eye(embeddings.shape[1])  # Identity matrix as placeholder
    
    # Transform embeddings to semantic space
    semantic_embeddings = embeddings @ transformation_matrix
    query_semantic = semantic_embeddings[query_idx].reshape(1, -1)
    
    # Use cosine similarity in transformed space
    query_tensor = torch.tensor(query_semantic, dtype=torch.float32)
    embeddings_tensor = torch.tensor(semantic_embeddings, dtype=torch.float32)
    sims = F.cosine_similarity(query_tensor, embeddings_tensor, dim=1).numpy()
    
    ranked = sorted(
        zip(products, sims),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    print("\n🔎 Learned Semantic Similarity Results:")
    for prod, score in ranked:
        print(f"Product ID: {prod['product_id']} | Path: {prod['path']} | Score: {score:.4f}")
    
    return ranked


def compute_feature_weighted_similarity(embeddings: np.ndarray, products: list, query_idx: int = 0, top_k: int = 5, feature_weights: np.ndarray = None):
    """
    Semantic similarity with feature importance weighting for CLIP embeddings.
    """
    if feature_weights is None:
        # For CLIP, you might want to weight different semantic regions differently
        feature_weights = np.ones(embeddings.shape[1])
    
    # Apply feature weights
    weighted_embeddings = embeddings * feature_weights
    query_weighted = weighted_embeddings[query_idx].reshape(1, -1)
    
    # Use cosine similarity on weighted embeddings
    query_tensor = torch.tensor(query_weighted, dtype=torch.float32)
    embeddings_tensor = torch.tensor(weighted_embeddings, dtype=torch.float32)
    sims = F.cosine_similarity(query_tensor, embeddings_tensor, dim=1).numpy()
    
    ranked = sorted(
        zip(products, sims),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    print("\n🔎 Feature-Weighted Semantic Similarity Results:")
    for prod, score in ranked:
        print(f"Product ID: {prod['product_id']} | Path: {prod['path']} | Score: {score:.4f}")
    
    return ranked


if __name__ == "__main__":
    dataset_path = "data"   # <-- put your dataset folder path here
    
    try:
        # Process only first 1000 images
        embeddings, products = build_embedding_matrix(dataset_path, max_images=1000)
        print(f"\nLoaded {len(products)} products with embeddings shape: {embeddings.shape}")
        
        # Example 1: Single query with CSV export
        print("\n" + "="*50)
        print("SINGLE QUERY EXAMPLE")
        top_results, all_results = compute_similarity(
            embeddings, products, 
            query_idx=0, top_k=5, method="cosine",
            export_to_csv=True, csv_filename="similarity_first_1000.csv"
        )
        
        # Example 2: CLIP-specific similarity with export
        print("\n" + "="*50)
        print("CLIP-OPTIMIZED EXAMPLE")
        clip_top, clip_all = compute_clip_specific_similarity(
            embeddings, products, 
            query_idx=0, top_k=5, export_to_csv=True
        )
        
        # Example 3: Export full similarity matrix (now feasible with 1000 images)
        print("\n" + "="*50)
        print("FULL SIMILARITY MATRIX")
        if len(products) <= 1000:  # Safe for 1000 images
            print("Computing full 1000x1000 similarity matrix...")
            matrix_df = export_comparison_matrix(
                embeddings, products, 
                method="cosine", 
                filename="similarity_matrix_1000.csv"
            )
        
        # Example 4: Batch processing first 10 queries
        print("\n" + "="*50)
        print("BATCH PROCESSING EXAMPLE")
        query_indices = list(range(min(10, len(products))))  # First 10 images as queries
        batch_results = batch_export_similarities(
            embeddings, products, 
            query_indices=query_indices, 
            method="cosine",
            output_dir="batch_similarity_1000"
        )
        
        print(f"\n✅ Processing complete! Analyzed {len(products)} images.")
        
    except Exception as e:
        print(f"Error: {e}")