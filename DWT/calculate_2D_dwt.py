import numpy as np
import calculate_dwt


def calc_dwt_gray(img, max_level, min_level, th, flag):
    dwt_coeffs = np.array(img)
    for i in range(max_level, min_level - 1, -1):
        coeffs_row = []
        # Apply DWT on the rows
        for row in dwt_coeffs:
            coeffs_row.append(calculate_dwt.calculate_dwt(row, i, th, flag))
        dwt_coeffs = np.array(coeffs_row)

        coeffs_row = []
        # Apply DWT on the columns
        for row in dwt_coeffs.T:
            coeffs_row.append(calculate_dwt.calculate_dwt(row, i, th, flag))
        dwt_coeffs = np.array(coeffs_row).T
        # Current version of the image array after each level of decomposition
    return dwt_coeffs
