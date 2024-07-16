import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to a common size
            images.append(img)
    return images

def extract_features(images):
    features = []
    for img in images:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute HOG features
        hog = cv2.HOGDescriptor()
        h = hog.compute(gray)
        
        features.append(h.flatten())
    return features

def cluster_images(features, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    labels = kmeans.labels_
    return labels

def organize_images_by_cluster(images, labels):
    clustered_images = defaultdict(list)
    for idx, label in enumerate(labels):
        clustered_images[label].append(images[idx])
    return clustered_images

folder = '../../pic/'
images = load_images_from_folder(folder)
features = extract_features(images)
num_clusters = 5
labels = cluster_images(features, num_clusters)
clustered_images = organize_images_by_cluster(images, labels)

for cluster, imgs in clustered_images.items():
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(imgs):
        plt.subplot(5, 5, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.suptitle(f'Cluster {cluster}')
    plt.show()
