# %%
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from skimage import data, img_as_float
import matplotlib.pyplot as plt

from utils.gac import gac
from utils.thresholding import thresholding, plot_thresholding
from utils.sift import grey_sift


img = cv.imread('../imgs/img_multiple.png', cv.IMREAD_GRAYSCALE)
image = img_as_float(img)

# %% Plot a histogram of the intensities in the image
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.hist(image.ravel(), bins=256)
ax.set_title("Histogram of the image")
ax.set_xlabel("Intensity")
ax.set_ylabel("Number of pixels")
plt.show()

# %% Thresholding
# Threshold for holes
HOLES = 0.15

plot_thresholding(image, HOLES)


# %% GAC
ls = gac(image)

fig = plt.figure(figsize=(8, 8))

plt.imshow(image, cmap="gray")
plt.axis('off')
plt.contour(ls, [0.5], colors='r')
plt.title("Morphological GAC segmentation", fontsize=12)

plt.show()


# %%

# Clusterize the pixels in the image with 3 groups
clt = KMeans(n_clusters=3)
clt.fit(image.reshape(-1, 1))
labels = clt.labels_
clustered_image = labels.reshape(image.shape)

# Plot the results of clustering and the original image
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax1.imshow(clustered_image)
ax1.set_title("Clusterized image")
ax1.axis('off')
ax2 = fig.add_subplot(122)
ax2.imshow(image, cmap="gray")
ax2.set_title("Original image")
ax2.axis('off')


# %%
# # Morphological ACWE
# image = img_as_float(img)

# # Initial level set
# init_ls = checkerboard_level_set(image.shape, 6)
# # List with intermediate results for plotting the evolution
# evolution = []
# callback = store_evolution_in(evolution)
# ls = morphological_chan_vese(image, num_iter=30, init_level_set=init_ls,
#                              smoothing=3, iter_callback=callback)

# fig = plt.figure(figsize=(8, 8))

# plt.imshow(image, cmap="gray")
# plt.axis('off')
# plt.contour(ls, [0.5], colors='r')
# plt.title("Morphological ACWE segmentation", fontsize=12)


# %%


# %%
