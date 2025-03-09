from sklearn.cluster import KMeans
import numpy as np

def cluster_images(image_embeddings, num_clusters=5):
    filenames = list(image_embeddings.keys())
    embeddings = np.vstack([image_embeddings[fn] for fn in filenames])
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    clusters = {i: [] for i in range(num_clusters)}
    for filename, label in zip(filenames, kmeans.labels_):
        clusters[label].append(filename)
    return clusters
