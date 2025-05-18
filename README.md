# Holographic Image Postprocessing


## Overview

This project focuses on accurate particle size estimation from 3D holographic images by leveraging a physics-based diffraction model and a deep learning pipeline. We reconstruct depth-resolved holographic images using the **Kirchhoff-Fresnel diffraction integral** and use a fine-tuned **ResNet50** model to predict particle sizes with high precision.

By combining principles of optical physics and computer vision, this approach achieves a **15% improvement in size estimation accuracy** over traditional methods and predicts particle sizes within **0.5 microns** on a dataset of **10,000+ holograms**.



## Key Features

*  **Physics-based 3D reconstruction** using Kirchhoff-Fresnel diffraction integral.
*  **Particle size estimation** with reduced Mean Absolute Error (MAE) by 20%.
*  **ResNet50** fine-tuned in TensorFlow with customized hyperparameter optimization.
*  Handles **10,000+ holographic samples** with enhanced computational efficiency.
* Provides **micron-level precision** in particle size predictions.



##  Methodology

### 1. Hologram Preprocessing

* Raw holographic images are normalized and filtered to remove background noise.
* Data augmentation is applied to improve model generalization.

### 2. 3D Reconstruction

* Implemented Kirchhoff-Fresnel diffraction integral to reconstruct depth slices from holograms.
* Generated focus-stacked 3D volumes to capture accurate particle morphology.

### 3. Feature Extraction and Model Training

* Extracted 2D slices and depth cues for model input.
* Fine-tuned a pre-trained **ResNet50** model using TensorFlow.
* Employed grid search for hyperparameter tuning (learning rate, batch size, optimizer, etc.).
* Used MAE and RMSE as evaluation metrics for performance tracking.



##  Results

| Metric                | Before Optimization | After Optimization |
| --------------------- | ------------------- | ------------------ |
| MAE                   | 1.25 microns        | **1.00 microns**   |
| Accuracy Improvement  | —                   | **+15%**           |
| Size Prediction Error | ±1.0 microns        | **±0.5 microns**   |



##  Tools & Libraries

* Python
* TensorFlow / Keras
* NumPy, SciPy
* OpenCV
* Matplotlib / Seaborn
* Custom implementation of Kirchhoff-Fresnel integral



##  Dataset

* Contains over **10,000 holographic images**.
* Each sample includes:

  * Raw hologram
  * Ground truth particle size (in microns)
  * Reconstructed depth slices

*(Dataset available upon request or according to data-sharing policy)*



##  Future Work

* Extend to multi-particle detection and segmentation.
* Integrate with real-time holographic imaging systems.
* Experiment with transformer-based architectures for 3D data understanding.



##  Contact

For questions or collaboration inquiries, feel free to reach out:

**Madhumitha Katam**
 \[[katammadhumitha@gmail.com](mailto:katammadhumitha@gmail.com)]
 \[[LinkedIn URL](https://www.linkedin.com/in/madhumithakatam/)]


