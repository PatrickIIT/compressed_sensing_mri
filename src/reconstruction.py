# src/reconstruction.py

"""
Implementation of classical reconstruction algorithms for Compressed Sensing MRI.

This module provides:
- Zero-Filled Reconstruction: The baseline method.
- ISTA: Iterative Shrinkage-Thresholding Algorithm with wavelet sparsity.
"""

import numpy as np
import pywt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def zero_filled_reconstruction(undersampled_kspace):
    """
    Performs a zero-filled reconstruction from undersampled k-space.

    Args:
        undersampled_kspace (np.ndarray): The complex-valued undersampled k-space data.

    Returns:
        np.ndarray: The reconstructed magnitude image.
    """
    return np.abs(ifft2(ifftshift(undersampled_kspace)))

def _soft_threshold(x, threshold):
    """Helper function for soft-thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def _wavelet_forward(image, wavelet='db4', level=3):
    """Helper function for forward wavelet transform."""
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices

def _wavelet_inverse(arr, coeff_slices, wavelet='db4'):
    """Helper function for inverse wavelet transform."""
    coeffs_from_arr = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(coeffs_from_arr, wavelet=wavelet)

def ista_wavelet_cs(undersampled_kspace, mask, initial_image=None, n_iters=10, lambda_val=0.0001):
    """
    Performs CS-MRI reconstruction using the ISTA with wavelet sparsity.

    Args:
        undersampled_kspace (np.ndarray): The complex undersampled k-space.
        mask (np.ndarray): The boolean sampling mask.
        initial_image (np.ndarray, optional): An initial image to start the iteration. 
                                              If None, uses zero-filled recon.
        n_iters (int): The number of iterations to perform.
        lambda_val (float): The regularization parameter.

    Returns:
        np.ndarray: The reconstructed magnitude image, clipped to [0, 1].
    """
    if initial_image is None:
        initial_image = zero_filled_reconstruction(undersampled_kspace)
        
    x_recon = initial_image.copy().astype(np.complex128)
    step_size = 1.0  # Lipschitz constant for this operator is 1

    for i in range(n_iters):
        # 1. Gradient descent step (data consistency)
        current_k_space = fftshift(fft2(x_recon))
        k_space_error = (current_k_space * mask) - undersampled_kspace
        grad_data_term = ifft2(ifftshift(k_space_error * mask))
        x_intermediate = x_recon - step_size * grad_data_term

        # 2. Proximal operator for L1 norm in wavelet domain (soft-thresholding)
        x_intermediate_real = np.real(x_intermediate)
        coeffs_arr, coeff_slices = _wavelet_forward(x_intermediate_real)
        threshold = lambda_val * step_size
        
        # Do not threshold the approximation coefficients (low-frequency components)
        coeffs_list_form = pywt.wavedec2(x_intermediate_real, 'db4', level=3)
        approx_coeffs_size = coeffs_list_form[0].size
        
        coeffs_arr_thresh = coeffs_arr.copy()
        coeffs_arr_thresh[approx_coeffs_size:] = _soft_threshold(
            coeffs_arr[approx_coeffs_size:], threshold
        )
        
        # Inverse wavelet transform
        x_reconstructed_real = _wavelet_inverse(coeffs_arr_thresh, coeff_slices)
        x_recon = x_reconstructed_real.astype(np.complex128)
    
    return np.clip(np.real(x_recon), 0, 1)
