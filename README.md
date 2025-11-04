Chakma Numerals Recognition (ResNet50 + Transformer + XAI)

This repository contains the code for a hybrid deep learning model (ResNet50 + CBAM + Transformer) to classify Chakma numerals. The project is implemented in PyTorch and is designed to be run in a Google Colab environment.

The model achieves high accuracy and includes Explainable AI (XAI) techniques like Grad-CAM and CBAM attention visualization to interpret its predictions.

 Key Features
Hybrid Architecture: Uses a ResNet50 backbone for feature extraction and a Transformer Encoder for classification.
Attention: Integrates a CBAM (Convolutional Block Attention Module) to refine features.
Explainable AI (XAI): Generates Grad-CAM heatmaps, CBAM spatial attention masks, and feature importance maps to understand model decisions.
Comprehensive Evaluation: Automatically generates and saves training curves, a confusion matrix, a full classification report, and ROC/AUC curves for each class.

🚀 How to Run this Project

This notebook is designed to be run on **Google Colab** as it requires a GPU and is pre-configured to use Google Drive for data storage.

1. Setup Your Data in Google Drive

Before you can run the notebook, you must place your dataset in your Google Drive at the following exact path:

`/content/drive/MyDrive/Chakma Numerals/`

The code expects your dataset to be structured in subfolders by class:
/content/drive/MyDrive/Chakma Numerals/ ├── 0/ │ ├── img1.png │ └── ... ├── 1/ │ ├── img2.png │ └── ... └── 9/
2. Run in Google Colab

    1.  Open [Google Colab](https://colab.research.google.com/).
    2.  Go to File > Open notebook... and select the GitHub tab.
    3.  Paste the URL of this repository and select `ResNet50.ipynb`.
    4.  Change the runtime to a GPU (Runtime > Change runtime type > T4 GPU).
    5.  Run the first cell to mount your Google Drive.
    6.  Run the rest of the cells to train the model and generate results.

3. View Results

xai_visualizations.png
<img width="7500" height="4500" alt="xai_visualizations" src="https://github.com/user-attachments/assets/d638d96f-cda6-499b-8ca2-3a6d1969c6d2" />

confusion_matrix.png 
<img width="3000" height="2400" alt="confusion_matrix" src="https://github.com/user-attachments/assets/1f1ffa09-0115-4fc6-852f-2e21f61e8e4f" />

roc_curve.png
<img width="3000" height="2400" alt="roc_curve" src="https://github.com/user-attachments/assets/cf072b9a-c9fa-475e-8b4f-186957e4834e" />

raining_curves.png
<img width="3600" height="1500" alt="training_curves" src="https://github.com/user-attachments/assets/d517f555-238d-4ec7-8d35-b5bec30112d8" />

🛠️ Libraries & Dependencies

This project requires the following major libraries:

 `torch` & `torchvision`
 `scikit-learn`
 `opencv-python`
 `seaborn` & `matplotlib`
 `pandas` & `numpy`
 `einops`
