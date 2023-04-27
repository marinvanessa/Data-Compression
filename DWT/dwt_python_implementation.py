import pywt
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('image/lena.bmp', cv2.IMREAD_GRAYSCALE)

# Set the number of decomposition levels
num_levels = 3

# Perform DWT with Haar wavelets and multiple levels of decomposition
coeffs = pywt.wavedec2(img, 'haar', level=num_levels)

# Create a 2D grid of subplots to display the images
fig, axes = plt.subplots(num_levels + 1, 3, figsize=(10, 10))

# Display the approximation coefficient image
axes[0, 0].imshow(coeffs[0].astype(np.uint8), cmap='gray')
axes[0, 0].set_title('Approximation Coefficient')

# Display the detail coefficient images at each level of decomposition
for i in range(num_levels):
    axes[i+1, 0].imshow(coeffs[i + 1][0].astype(np.uint8), cmap='gray')
    axes[i+1, 0].set_title(f'Horizontal Detail Coefficient (level {i + 1})')

    axes[i+1, 1].imshow(coeffs[i + 1][1].astype(np.uint8), cmap='gray')
    axes[i+1, 1].set_title(f'Vertical Detail Coefficient (level {i + 1})')

    axes[i+1, 2].imshow(coeffs[i + 1][2].astype(np.uint8), cmap='gray')
    axes[i+1, 2].set_title(f'Diagonal Detail Coefficient (level {i + 1})')

# Set the spacing between subplots
plt.subplots_adjust(hspace=1.5, wspace=1.3)

# Show the plot
plt.show()
