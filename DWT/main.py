import cv2
import matplotlib.pyplot as plt
import numpy as np

import calculate_2D_dwt

img = cv2.imread('image/lena.bmp')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_arr = np.array(img)


th = 2
thFlag = True
max_level = 9
min_level = 9

image_after_DWT = calculate_2D_dwt.calc_dwt_gray(img, max_level, min_level, th, thFlag)

original_size = img.shape[0] * img_arr.shape[1]
compressed_size = image_after_DWT.shape[0] * image_after_DWT.shape[1]

compression_ratio = original_size / compressed_size
print(original_size)
print(compressed_size)
print(compression_ratio)


fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(image_after_DWT)
ax[1].set_title('DWT Haar')
ax[1].axis('off')

plt.show()

