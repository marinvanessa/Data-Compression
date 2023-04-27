import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('image/lena.bmp', cv2.IMREAD_GRAYSCALE)

# Compute the DCT
dct = cv2.dct(np.float32(img))

# Define the quantization level
quant_level = 10

# Quantize each DCT coefficient
quant_dct = np.round(dct / quant_level)

# Apply inverse DCT to obtain the quantized image
quant_img = cv2.idct(quant_dct)

# Calculate original and compressed sizes
original_size = img.size * img.itemsize
compressed_size = quant_dct.size * quant_dct.itemsize

# Display the original and quantized images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')

compression_ratio = float(original_size) / float(compressed_size)
axs[1].imshow(quant_img, cmap='gray')
axs[1].set_title('Quantized Image (Compression Ratio: {:.2f})'.format(compression_ratio))

plt.show()
