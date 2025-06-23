# src/masks.py

"""
Functions for generating k-space undersampling masks for Compressed Sensing MRI.

This module provides implementations for:
- 1D Cartesian undersampling
- 2D Random undersampling
- Variable-Density polynomial undersampling
- Edge-Enhanced (anatomy-aware) undersampling
"""

import numpy as np
from skimage.filters import sobel
from numpy.fft import fft2, fftshift

def create_cartesian_mask(shape, acceleration_factor, center_fraction=0.08):
    """
    Creates a 1D Cartesian undersampling mask.

    Args:
        shape (tuple): The shape of the k-space (height, width).
        acceleration_factor (int): The target acceleration factor.
        center_fraction (float): Fraction of the central k-space to fully sample.

    Returns:
        np.ndarray: A boolean mask of the specified shape.
    """
    num_cols = shape[1]
    num_low_freqs = int(round(num_cols * center_fraction))
    
    mask = np.zeros(shape, dtype=bool)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[:, pad : pad + num_low_freqs] = True
    
    num_sampled_lines_outer = (num_cols - num_low_freqs) // acceleration_factor
    outer_lines_indices = np.setdiff1d(np.arange(num_cols), np.arange(pad, pad + num_low_freqs))
    
    permuted_outer_lines = np.random.permutation(outer_lines_indices)
    selected_outer_lines = permuted_outer_lines[:num_sampled_lines_outer]
    mask[:, selected_outer_lines] = True
    
    return mask

def create_random_2d_mask(shape, acceleration_factor, center_fraction=0.08, seed=None):
    """
    Creates a 2D random undersampling mask with a fully-sampled center.

    Args:
        shape (tuple): The shape of the k-space.
        acceleration_factor (int): The target acceleration factor.
        center_fraction (float): Fraction of the central k-space area to fully sample.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: A boolean mask.
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_points = int(np.prod(shape) / acceleration_factor)
    mask = np.zeros(shape, dtype=bool)
    
    # Fully sample center
    center_rows = int(shape[0] * center_fraction)
    center_cols = int(shape[1] * center_fraction)
    r_start, r_end = shape[0]//2 - center_rows//2, shape[0]//2 + center_rows//2
    c_start, c_end = shape[1]//2 - center_cols//2, shape[1]//2 + center_cols//2
    mask[r_start:r_end, c_start:c_end] = True
    
    num_sampled_center = np.sum(mask)
    remaining_points = num_points - num_sampled_center

    if remaining_points > 0:
        outer_indices = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                if not (r_start <= r < r_end and c_start <= c < c_end):
                    outer_indices.append((r, c))
        
        if len(outer_indices) > 0:
            chosen_indices_flat = np.random.choice(len(outer_indices), 
                                                     min(remaining_points, len(outer_indices)), 
                                                     replace=False)
            for flat_idx in chosen_indices_flat:
                r, c = outer_indices[flat_idx]
                mask[r, c] = True
                
    return mask

def create_variable_density_mask(shape, acceleration_factor, center_fraction=0.08, poly_degree=2, seed=None):
    """
    Creates a variable-density random mask using a polynomial PDF.

    Args:
        shape (tuple): The shape of the k-space.
        acceleration_factor (int): The target acceleration factor.
        center_fraction (float): Fraction of the central k-space to fully sample.
        poly_degree (int): The degree of the polynomial for PDF decay.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: A boolean mask.
    """
    if seed is not None:
        np.random.seed(seed)
        
    rows, cols = shape
    center_x, center_y = cols // 2, rows // 2
    x_coords = np.abs(np.arange(cols) - center_x)
    y_coords = np.abs(np.arange(rows) - center_y)
    dist_x, dist_y = np.meshgrid(x_coords, y_coords)
    
    norm_dist_x = dist_x / (np.max(dist_x) if np.max(dist_x) > 0 else 1)
    norm_dist_y = dist_y / (np.max(dist_y) if np.max(dist_y) > 0 else 1)
    
    pdf = (1 - norm_dist_x**poly_degree) * (1 - norm_dist_y**poly_degree)
    
    target_samples = int(np.prod(shape) / acceleration_factor)
    flat_pdf = pdf.flatten()
    sorted_indices = np.argsort(-flat_pdf) # Sort descending
    
    mask = np.zeros(shape, dtype=bool).flatten()
    mask[sorted_indices[:target_samples]] = True
    mask = mask.reshape(shape)

    # Ensure center is fully sampled
    r_start = rows//2 - int(rows*center_fraction)//2
    c_start = cols//2 - int(cols*center_fraction)//2
    mask[r_start:r_start+int(rows*center_fraction), 
         c_start:c_start+int(cols*center_fraction)] = True
         
    return mask

def create_edge_enhanced_mask(shape, acceleration_factor, reference_image, center_fraction=0.08, seed=None):
    """
    Creates an anatomy-aware mask based on the edges of a reference image.

    Args:
        shape (tuple): The shape of the k-space.
        acceleration_factor (int): The target acceleration factor.
        reference_image (np.ndarray): A 2D ground truth image to derive edge information.
        center_fraction (float): Fraction of the central k-space to fully sample.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: A boolean mask.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Edge detection
    edge_map = sobel(reference_image)
    if np.max(edge_map) > 0:
        edge_map /= np.max(edge_map)
    
    # 2. Get k-space energy of edges
    k_space_energy = np.abs(fftshift(fft2(edge_map)))
    if np.max(k_space_energy) > 0:
        k_space_energy /= np.max(k_space_energy)
    
    # 3. Create a center-weighted PDF
    rows, cols = shape
    center_x, center_y = cols // 2, rows // 2
    x_coords = np.abs(np.arange(cols) - center_x) / (cols / 2)
    y_coords = np.abs(np.arange(rows) - center_y) / (rows / 2)
    dist_x, dist_y = np.meshgrid(x_coords, y_coords)
    center_pdf = (1 - dist_x**2) * (1 - dist_y**2)
    
    # 4. Combine PDFs and sample
    edge_weight = 0.7
    combined_pdf = edge_weight * k_space_energy + (1 - edge_weight) * center_pdf
    
    # Use the same sampling logic as variable density
    target_samples = int(np.prod(shape) / acceleration_factor)
    flat_pdf = combined_pdf.flatten()
    sorted_indices = np.argsort(-flat_pdf)
    
    mask = np.zeros(shape, dtype=bool).flatten()
    mask[sorted_indices[:target_samples]] = True
    mask = mask.reshape(shape)

    # Ensure center is fully sampled
    r_start = rows//2 - int(rows*center_fraction)//2
    c_start = cols//2 - int(cols*center_fraction)//2
    mask[r_start:r_start+int(rows*center_fraction), 
         c_start:c_start+int(cols*center_fraction)] = True
         
    return mask
