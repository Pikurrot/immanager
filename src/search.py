import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_images(text_embedding, image_embeddings, top_k=5):
    similarities = {}
    for filename, img_embedding in image_embeddings.items():
        sim = cosine_similarity(text_embedding, img_embedding)
        similarities[filename] = sim
    # Sort based on similarity score in descending order
    sorted_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_images[:top_k]
