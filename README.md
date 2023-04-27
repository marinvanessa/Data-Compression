# Data-Compression
The project involves image compression using the Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT) algorithms in Python.

The first step is to load an image in grayscale using OpenCV. Then, the DWT algorithm is applied to the image using the existing implementation in Python.

Then, DCT algorithm is applied to the image.

After applying the DCT algorithm, the image is quantized using a specified quantization level. The quantization process involves dividing the DCT coefficients by the quantization level and rounding off the result to the nearest integer.

Finally, the compressed image is obtained by applying the inverse DCT algorithm to the quantized coefficients.

In addition to utilizing the pre-existing DWT and DCT algorithms available in Python, I took the initiative to implement my own custom DWT algorithm as part of the project.
