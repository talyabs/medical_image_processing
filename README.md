# ViT for AD Detection with CDSS

## Overview

This project focuses on leveraging machine learning techniques to diagnose Alzheimer's disease using MRI scans from the OASIS dataset. Our work centers around training deep learning models, specifically convolutional neural networks (CNNs) and Vision Transformers (ViTs), to classify patients into categories such as "Dementia" and "Non-Dementia." A key component of the project is the integration of explainability through a Clinical Decision Support System (CDSS), which provides visual reports to help clinicians understand how the model arrives at its predictions.

In high-stakes medical domains, explainability is crucial. Therefore, we have implemented several methods, including LIME explanations, saliency maps, and uncertainty estimation (Monte Carlo Dropout), to enhance the transparency of the model’s decision-making process. The goal is to build trust with clinicians by allowing them to verify that the model is focusing on relevant brain regions affected by Alzheimer's when making predictions.

Our models are evaluated on binary and multiclass classification tasks, with a focus on interpretability to ensure that the system is both accurate and useful for clinical decision-making.

For a detailed explanation of the models, methods, and results, refer to the full project report:  
[Full Report](https://drive.google.com/file/d/1plcKaKjirVqjsOg2Snph4bWNQPlCIZw0/view?usp=sharing)

### Project Based On:
This project is inspired by the paper *Designing a clinical decision support system for Alzheimer’s diagnosis on OASIS-3 data set*, authored by **Farzaneh Salami**, **Ali Bozorgi-Amiri**, **Reza Tavakkoli-Moghaddam**, **Ghulam Mubashar Hassan**, and **Amitava Datta**. It builds on the methods outlined in the paper to develop a similar CDSS for Alzheimer's diagnosis using the OASIS dataset.

## Project Structure

- **code/**: Contains all the scripts to train the different models, including:
    - `train_vit_binary.py`: Script to train the binary Vision Transformer model.
    - `train_vit.py`: Script to train the multiclass Vision Transformer model.
    - `train_resnet.py`, `train_densenet.py`, `train_inceptionv3.py`: Scripts for training various CNN-based models (ResNet, DenseNet, InceptionV3).
    - `ensemble_model.py`: Script to train an ensemble model, combining multiple model architectures for improved performance.
    - `cdss.py`: Script to generate the Clinical Decision Support System (CDSS) reports.
    - `models_utils.py`: Helper functions for model training and utilities.

- **notebooks/**: Contains Jupyter notebooks used for data exploration and analysis:
    - `Oasis_Alzheimer_Diagnosis.ipynb`: Notebook with Exploratory Data Analysis (EDA) of the dataset and initial model experimentation.

- **test_data/**: Contains the list of filenames for the test dataset used in CDSS report generation:
    - `test_filenames.csv`: Lists the filenames of the test users for which CDSS reports are generated.

- **plots/**: Contains evaluation plots and results from model training:
    - `final_vit_binary_plots`: Plots and metrics from the final Vision Transformer binary model evaluation.
    - `multiclass_plots`: Plots from multiclass classification tasks.

- **cdss_reports/**: Contains generated CDSS reports that explain the model’s predictions:
    - Example reports, such as:
      - `cdss_report_patient_0006.png`: A correct prediction report where both the true and predicted class are "Non-Dementia".
      - `cdss_report_patient_0069.png`: A report where the model predicted incorrectly, showing "Non-Dementia" as the predicted class but the true class is "Dementia".

## CDSS Reports

The **Clinical Decision Support System (CDSS)** generates reports explaining the model’s predictions. These reports include:
- The original MRI image for context.
- The predicted class (Dementia or Non-Dementia).
- LIME explanations highlighting areas of the image important to the model's prediction.
- Saliency maps visualizing regions that contributed the most to the model’s decision.
- Probability scores and uncertainty estimates to give clinicians an understanding of the confidence behind the predictions.

### Examples:
- **Correct Prediction**: [`cdss_report_patient_0006.png`](./cdss_reports/cdss_report_patient_0006.png) (True class: Non-Dementia, Predicted class: Non-Dementia)
- **Incorrect Prediction**: [`cdss_report_patient_0069.png`](./cdss_reports/cdss_report_patient_0069.png) (True class: Dementia, Predicted class: Non-Dementia)


## Getting Started

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd medical_image_processing
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the models or generate CDSS reports:
    ```bash
    python code/train_vit_binary.py
    ```

4. Generate CDSS reports:
    ```bash
    python code/cdss.py
    ```

