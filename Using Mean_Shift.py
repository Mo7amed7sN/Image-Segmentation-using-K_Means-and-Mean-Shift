import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift

# Load the Img..
img = cv2.imread("Images/dog.4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

# Reshape img to give it to estimate_bandwidth function in sklearn.cluster.
pixels = img.reshape((-1, 3))
pixels = np.float32(pixels)
print(pixels.shape)

# Calc the Bandwidth..
bandwidth = estimate_bandwidth(pixels, quantile=.1,n_samples=500)

# MeanShift Instance from sklearn..
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(pixels)

# Get labels and Centroids..
labels = ms.labels_
cluster_centers = ms.cluster_centers_
cluster_centers = np.uint8(cluster_centers)
labels = labels.flatten()

# Get segmented img..
segmented_img = cluster_centers[labels.flatten()]
segmented_img = segmented_img.reshape(img.shape)
plt.imshow(segmented_img)
plt.show()
