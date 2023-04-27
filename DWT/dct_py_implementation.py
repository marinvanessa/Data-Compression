import cv2
import pywt
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('image/lena.bmp', cv2.IMREAD_GRAYSCALE)

# Compute the DCT at levels 1, 2,... and 9
dct1 = pywt.dwt2(img, 'haar')[0]
dct2 = pywt.dwt2(dct1, 'haar')[0]
dct3 = pywt.dwt2(dct2, 'haar')[0]
dct4 = pywt.dwt2(dct3, 'haar')[0]
dct5 = pywt.dwt2(dct4, 'haar')[0]
dct6 = pywt.dwt2(dct5, 'haar')[0]
dct7 = pywt.dwt2(dct6, 'haar')[0]
dct8 = pywt.dwt2(dct7, 'haar')[0]
dct9 = pywt.dwt2(dct8, 'haar')[0]

# Plot the original image and the DCT coefficients at each level
plt.subplot(1, 10, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 10, 2)
plt.imshow(dct1, cmap='gray')
plt.title('DCT Level 1')

plt.subplot(1, 10, 3)
plt.imshow(dct2, cmap='gray')
plt.title('DCT Level 2')

plt.subplot(1, 10, 4)
plt.imshow(dct3, cmap='gray')
plt.title('DCT Level 3')

plt.subplot(1, 10, 5)
plt.imshow(dct4, cmap='gray')
plt.title('DCT Level 4')

plt.subplot(1, 10, 6)
plt.imshow(dct5, cmap='gray')
plt.title('DCT Level 5')

plt.subplot(1, 10, 7)
plt.imshow(dct6, cmap='gray')
plt.title('DCT Level 6')

plt.subplot(1, 10, 8)
plt.imshow(dct7, cmap='gray')
plt.title('DCT Level 7')

plt.subplot(1, 10, 9)
plt.imshow(dct8, cmap='gray')
plt.title('DCT Level 8')

plt.subplot(1, 10, 10)
plt.imshow(dct9, cmap='gray')
plt.title('DCT Level 9')

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.4, hspace=1.5)

plt.show()
