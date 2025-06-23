# Comparative Analysis of Compressed Sensing Techniques for MRI Reconstruction

This repository contains the code and experimental logs for a comprehensive project on accelerating Magnetic Resonance Imaging (MRI) using Compressed Sensing (CS). The research systematically compares the performance of a classical iterative algorithm (ISTA) against a custom-developed Convolutional Neural Network (CNN) across various k-space undersampling patterns.

---

## 🚀 Key Finding

The most significant finding of this research is that the **k-space sampling strategy is a more dominant factor in reconstruction quality than the choice between classical iterative algorithms and deep learning models**, especially in data-limited scenarios.

Our experiments consistently demonstrate that a classical **ISTA algorithm combined with physics-informed sampling masks (Variable Density and Edge-Enhanced) significantly outperforms a custom U-Net CNN** trained on the same undersampled data. This underscores the critical importance of intelligent data acquisition in the CS-MRI pipeline.

---

## 🧠 Project Overview

The core objective of this project is to dissect the interplay between data acquisition (sampling) and image reconstruction in CS-MRI. We investigate whether a well-established iterative algorithm can compete with a modern deep learning approach if provided with intelligently sampled data.

### Features & Methodology

*   **Datasets Analyzed:**
    *   **OASIS-1:** T1-weighted brain MRI scans.
    *   **UC Berkeley Knee Dataset:** High-resolution 3D FSE knee scans with raw k-space data.
    *   **fastMRI Brain Dataset:** Multi-coil raw k-space data for brain MRI.

*   **K-Space Undersampling Masks (R=4):**
    1.  `1D Cartesian`: Standard uniform undersampling.
    2.  `2D Random`: Incoherent point-wise sampling.
    3.  `Variable-Density`: Physics-informed polynomial mask prioritizing the k-space center.
    4.  `Edge-Enhanced`: Anatomy-aware mask that prioritizes k-space regions corresponding to anatomical edges.

*   **Reconstruction Algorithms:**
    *   **ISTA:** A classical Iterative Shrinkage-Thresholding Algorithm with Wavelet sparsity.
    *   **CNN:** A custom U-Net-based deep learning model implemented in PyTorch, trained for the reconstruction task.

*   **Evaluation Metrics:**
    *   Peak Signal-to-Noise Ratio (PSNR)
    *   Structural Similarity Index (SSIM)

---

## 📊 Results Highlights

Across all datasets, the Variable Density and Edge-Enhanced masks provided a vastly superior starting point for reconstruction, enabling ISTA to achieve exceptional results.

**Performance on UC Berkeley Knee Dataset (R=4, Variable Density Mask)**

| Method            | PSNR (dB) | SSIM   | Time (s)           |
| ----------------- | --------- | ------ | ------------------ |
| Zero-Filled       | 40.92     | 0.9613 | 0.00               |
| **ISTA (10 iter.)** | **40.93** | **0.9614** | 0.43               |
| CNN (U-Net)       | 31.86     | 0.8056 | 185.01 (Train+Inf) |

**Visual Comparison of Reconstructions:**

![Visual Results Comparison](path/to/your/result_image.png)
*(Note: Replace `path/to/your/result_image.png` with a path to one of your visual comparison figures, like the one from Experiment 9.)*

---

## 📂 Repository Structure

The project is organized to separate the experimental logs from the core source code.

```
compressed_sensing_mri/
│
├── experiments/
│   ├── EA_PROJECT_EXPERIMENTS.txt  # Detailed logs for all 10 experiments
│   └── ... (Individual experiment notebooks or scripts can be placed here)
│
├── src/
│   ├── data_loader.py              # Scripts for loading datasets
│   ├── masks.py                    # Functions for generating sampling masks
│   ├── models.py                   # CNN (U-Net) architecture
│   └── reconstruction.py           # ISTA implementation
│
├── README.md                       # You are here
└── requirements.txt                # Project dependencies
```

---

## 🛠️ Setup and Installation

To set up the environment and run the experiments, please follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PatrickIIT/compressed_sensing_mri.git
    cd compressed_sensing_mri
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    numpy
    torch
    torchvision
    scikit-image
    matplotlib
    pywavelets
    pydicom
    nibabel
    h5py
    ```

---

## 🔬 How to Run the Experiments

The detailed logs and code snippets for all experiments are located in the `experiments/EA_PROJECT_EXPERIMENTS.txt` file.

To reproduce a specific experiment:
1.  Ensure you have downloaded the required datasets (OASIS-1, UCB Knee, fastMRI) and placed them in an accessible directory.
2.  Open a Jupyter Notebook or create a Python script.
3.  Copy the code block corresponding to the desired experiment from the log file.
4.  **Important:** Update the `base_kaggle_input_path` or other dataset path variables to point to the location of your downloaded data.
5.  Execute the code.

---

## 💡 Discussion and Future Work

This study reveals that optimizing the data acquisition strategy can be more impactful than increasing the complexity of the reconstruction algorithm. A simple, mathematically sound method like ISTA can outperform a CNN when provided with high-quality undersampled data from a Variable Density mask.

Future research directions include:
*   **Hybrid Models:** Developing models that embed ISTA-like iterative steps within a deep neural network to get the best of both worlds.
*   **Learnable Sampling Masks:** Training a neural network to jointly optimize the k-space sampling pattern and the reconstruction algorithm.
*   **Clinical Validation:** Testing these methods on prospectively undersampled data from clinical scanners to evaluate performance against real-world noise and artifacts.

---

## 📜 Citation

If you use this work in your research, please cite it as follows:

```bibtex
@misc{vincent2024cs_mri_comparison,
  author       = {Vincent, Patrick},
  title        = {Comparative Analysis of Compressed Sensing Techniques for MRI Reconstruction},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/PatrickIIT/compressed_sensing_mri}}
}
```

---

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 📧 Contact

**Patrick Vincent**
*   **Institute:** Indian Institute of Technology Madras, Zanzibar Campus
*   **Email:** `zda24m007@iitmz.ac.in`
*   **GitHub:** [PatrickIIT](https://github.com/PatrickIIT)
