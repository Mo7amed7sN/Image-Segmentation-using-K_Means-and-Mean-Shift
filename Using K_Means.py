import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Img..
img = cv2.imread("Images/dog.4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

# Reshape img to give it to kmeans function in open_cv.
pixels = img.reshape((-1, 3))
pixels = np.float32(pixels)
print(pixels.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# using Kmeans in open_cv to get labels and centroids.
k = 2
skip, labels, (centroids) = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centroids = np.uint8(centroids)
labels = labels.flatten()

# Get the segmented img.
segmented_img = centroids[labels.flatten()]
segmented_img = segmented_img.reshape(img.shape)
plt.imshow(segmented_img)
plt.show()
