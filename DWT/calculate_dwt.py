import numpy as np


def calculate_dwt(img, level, th, flag=False):
    # Maximum level of decomposition is determined by the size of the input image,
    # and we need this in order to know how many times we can apply DWT on our image

    # For getting the correct maximum level of decomposition I will
    # take the log base 2 of the total number of elements in my image

    # Maximum level of decomposition
    max_level = np.log2(len(img))

    if (max_level - int(max_level)) > 0:
        max_level = np.ceil(max_level)

    # Getting the number of values needed to pad the input image to a length that is a power of 2,
    # which is necessary for the DWT algorithm to work.
    total_values = int(np.power(2, max_level))

    # Padding the input image with zeros to make its length a power of 2
    padded_img = np.zeros(total_values)
    padded_img[:len(img)] = img

    # Partition of padded_img
    # Image is divided in two parts: the first part will have the 2 power of maxLevel elements of the padded image
    # and the second part will have the remaining elements from it
    val = int(np.power(2, level))
    first = padded_img[0:val]
    second = padded_img[val:]

    # Calculate new values
    # Calculate the approximation coefficients and detail coefficients at a given level od decomposition
    # The approximation coefficients represent the low-frequency components of the image
    # The Detail coefficients represent the high-frequency components of the image
    approximation_coefficients = []
    detail_coefficients = []
    i = 0
    while i < len(first):
        aux = (first[i] + first[i + 1]) / 2
        if flag:
            if aux >= th:
                aux = aux
            else:
                aux = 0
        approximation_coefficients.append(aux)

        aux = (first[i] - first[i + 1]) / 2
        if flag:
            if aux >= th:
                aux = aux
            else:
                aux = 0
        detail_coefficients.append(aux)
        i += 2  # move to the next pair

    result = []
    result.extend(approximation_coefficients)  # first DWT array
    result.extend(detail_coefficients)  # second array DWT
    result.extend(second)  # remaining values

    return np.array(result)  # return the final DWT array

# Result array represents the complete DWT of the input image
# This array contains the DWT coefficients for all levels of the DWT,
# starting from low frequency components and progressing to the highest frequency
# components
