# src/data_loader.py

"""
Data loading and preprocessing utilities for various MRI datasets.

This module provides functions to load and process data from:
- OASIS-1 (NIfTI format)
- UC Berkeley Knee Dataset (DICOM and .cfl raw k-space)
- fastMRI Brain Dataset (HDF5 format)
"""

import os
import glob
import re
import numpy as np
import nibabel as nib
import pydicom
import h5py
from skimage.transform import resize
from numpy.fft import fft2, ifftshift

def load_oasis_slice(file_path, slice_idx, target_size=(256, 256)):
    """
    Loads a single 2D slice from an OASIS-1 NIfTI file.

    Args:
        file_path (str): Path to the .img or .nii file.
        slice_idx (int): The index of the axial slice to extract.
        target_size (tuple): The desired output size (height, width).

    Returns:
        np.ndarray: The normalized 2D image slice, or None if an error occurs.
    """
    try:
        nii = nib.load(file_path)
        img_data = nii.get_fdata()
        
        # Assuming axial slice is from the 3rd dimension
        img_slice = img_data[:, :, slice_idx, 0].astype(np.float32)
        
        if img_slice.shape != target_size:
            img_slice = resize(img_slice, target_size, anti_aliasing=True)
            
        if np.max(img_slice) > np.min(img_slice):
            img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
        else:
            img_slice = np.zeros_like(img_slice)
            
        return img_slice
    except Exception as e:
        print(f"Error loading OASIS file {file_path}: {e}")
        return None

def load_ucb_knee_slice(dicom_path, kspace_cfl_path, params_path, target_slice_idx=128):
    """
    Loads the ground truth and k-space for the UCB Knee dataset.
    It attempts to use the raw k-space data and falls back to DICOM if it fails.

    Args:
        dicom_path (str): Path to the .mag DICOM file.
        kspace_cfl_path (str): Path to the kspace.cfl file.
        params_path (str): Path to the params.txt file.
        target_slice_idx (int): The slice index to target (e.g., 128 for Sec_128.mag).

    Returns:
        tuple: (ground_truth_image, k_space_full) or (None, None) on failure.
    """
    # Load ground truth from DICOM
    dicom_data = pydicom.dcmread(dicom_path)
    ground_truth = dicom_data.pixel_array.astype(np.float32)
    ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8)

    # Attempt to load and process raw k-space
    try:
        with open(params_path, 'r') as f:
            params = f.read()
            rows = int(re.search(r'rhnframes yres:\s*(\d+)', params).group(1))
            cols = int(re.search(r'rhfrsize xres:\s*(\d+)', params).group(1))
            slices = int(re.search(r'rhnslices slices in a pass:\s*(\d+)', params).group(1))
        
        dims = (rows, cols, 8, slices) # Assume 8 coils
        raw_kspace_data = np.fromfile(kspace_cfl_path, dtype=np.float32)
        kspace_data = (raw_kspace_data[::2] + 1j * raw_kspace_data[1::2]).reshape(dims, order='F')

        # Combine coils using Root-Sum-of-Squares (RSS)
        kspace_slice = kspace_data[:, :, :, target_slice_idx]
        image_slice = ifft2(ifftshift(kspace_slice, axes=(0,1)), axes=(0,1))
        image_rss = np.sqrt(np.sum(np.abs(image_slice)**2, axis=2))
        
        # Zero-pad to match DICOM resolution
        padded_kspace = np.zeros_like(ground_truth, dtype=np.complex128)
        pad_x = (ground_truth.shape[0] - image_rss.shape[0]) // 2
        pad_y = (ground_truth.shape[1] - image_rss.shape[1]) // 2
        padded_kspace[pad_x:pad_x+rows, pad_y:pad_y+cols] = fftshift(fft2(image_rss))
        
        k_space_full = padded_kspace
        print("Successfully loaded and processed raw k-space data.")

    except Exception as e:
        print(f"Could not process raw k-space data ({e}). Falling back to simulating from DICOM.")
        k_space_full = fftshift(fft2(ground_truth))

    return ground_truth, k_space_full

def load_fastmri_slice(h5_path, slice_idx, target_size=(256, 256)):
    """
    Loads a single 2D slice from a fastMRI HDF5 file.

    Args:
        h5_path (str): Path to the .h5 file.
        slice_idx (int): The slice index to extract.
        target_size (tuple): The desired output size.

    Returns:
        np.ndarray: The normalized 2D image slice, or None if an error occurs.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            kspace_slice = f['kspace'][slice_idx] # (coils, height, width)
            img_coils = ifft2(ifftshift(kspace_slice, axes=(1, 2)), axes=(1, 2))
            img_rss = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
            
            img_rss = resize(img_rss, target_size, anti_aliasing=True).astype(np.float32)
            
            if np.max(img_rss) > np.min(img_rss):
                img_rss = (img_rss - np.min(img_rss)) / (np.max(img_rss) - np.min(img_rss))
            else:
                img_rss = np.zeros_like(img_rss)

            return img_rss
    except Exception as e:
        print(f"Error loading fastMRI file {h5_path}: {e}")
        return None
