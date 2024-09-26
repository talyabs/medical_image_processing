import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lime import lime_image
from models_utils import get_info_from_filename
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights

# Define image transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(VisionTransformerModel, self).__init__()
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.encoder.layers[-4:].parameters():
            param.requires_grad = True

        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.attention_maps = []

    def forward(self, x):
        return self.vit(x)

    def hook_attention(self):
        for layer in self.vit.encoder.layers:
            layer.self_attention.register_forward_hook(self.capture_attention)

    def capture_attention(self, module, input, output):
        self.attention_maps.append(output[1])  # Store attention weights


# Monte Carlo Dropout to estimate uncertainty
def monte_carlo_dropout(model, image_tensor, num_samples=50):
    model.train()
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(image_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            predictions.append(probabilities)
    return np.mean(predictions, axis=0), np.std(predictions, axis=0)


# Generate saliency map
def generate_saliency_map(model, image_tensor):
    image_tensor.requires_grad_()
    logits = model(image_tensor)
    logits.max().backward()
    saliency = image_tensor.grad.abs().squeeze().cpu().numpy()
    return np.max(saliency, axis=0)


def extract_true_class_from_path(file_path):
    """
    Extract the true class from the file path. Assumes the class name appears after 'Data/' and before the next '/'.
    For example: Data/Very mild Dementia/OAS1_0003_MR1_mpr-1_100.jpg
    """
    return file_path.split("/")[-2]


def get_info_from_filename(filepath):
    """
    Extract patient info from the filename.
    For example, filename: 'OAS1_0003_MR1_mpr-1_100.jpg'
    """
    filename = os.path.basename(filepath)
    pattern = re.compile(r"OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+).jpg")
    match = pattern.match(filename)

    if match:
        patient_id = match.group(1)
        mr_id = match.group(2)
        scan_id = match.group(3)
        layer_id = match.group(4)
        return patient_id, mr_id, scan_id, layer_id
    else:
        return None


def get_predicted_class(probabilities):
    """
    Return the predicted class based on the model's probabilities.
    Assumes binary classification (0: Non-Dementia, 1: Dementia).
    """
    predicted_class_idx = np.argmax(probabilities, axis=1)[0]
    return "Non-Dementia" if predicted_class_idx == 0 else "Dementia"


def generate_report(model, image_path, output_dir, device):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)  # Convert to numpy array for LIME

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get model predictions
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()

    # Extract true class from file path
    true_class = extract_true_class_from_path(image_path)

    # Get patient info from the filename
    patient_info = get_info_from_filename(image_path)
    if patient_info:
        patient_id, mr_id, scan_id, layer_id = patient_info
    else:
        patient_id, mr_id, scan_id, layer_id = (
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
        )

    # Get predicted class
    predicted_class = get_predicted_class(probabilities)

    # Initialize LIME image explainer
    explainer = lime_image.LimeImageExplainer()

    def classifier_fn(imgs):
        processed_imgs = []
        for img in imgs:
            if img.shape[-1] == 3:
                img = Image.fromarray(img.astype("uint8"))
                img_tensor = transform(img).unsqueeze(0)
                processed_imgs.append(img_tensor)

        batch_tensor = torch.cat(processed_imgs).to(device)
        with torch.no_grad():
            output = model(batch_tensor)

        return F.softmax(output, dim=1).cpu().numpy()

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image_np,
        classifier_fn=classifier_fn,
        top_labels=2,
        hide_color=0,
        num_samples=1000,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    img_boundaries = mark_boundaries(temp, mask)

    # Monte Carlo Dropout: Uncertainty estimation
    mean_preds, uncertainty = monte_carlo_dropout(model, image_tensor)

    # Generate saliency map
    saliency_map = generate_saliency_map(model, image_tensor)
    saliency_map = np.transpose(saliency_map, (1, 0))
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Display the file path and additional info on top as a title
    fig.suptitle(
        f"File Path: {image_path}\nTrue Class: {true_class}\nPredicted Class: {predicted_class}\n"
        f"Patient ID: {patient_id}, MR ID: {mr_id}, Scan ID: {scan_id}, Layer ID: {layer_id}",
        fontsize=12,
        ha="center",
    )

    # Plot original image
    ax[0, 0].imshow(image_np)
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Original Image")

    ax[0, 1].imshow(img_boundaries)
    ax[0, 1].axis("off")
    ax[0, 1].set_title("LIME Explanation")

    ax[1, 0].imshow(saliency_map, cmap="hot")
    ax[1, 0].axis("off")
    ax[1, 0].set_title("Saliency Map")

    ax[1, 1].text(
        0.5,
        0.5,
        f"**Probabilities**:\n"
        f" - Non-Dementia: {probabilities[0][0]:.2f} \n"
        f" - Dementia: {probabilities[0][1]:.2f} \n\n"
        f"**What does this mean?**:\n"
        f" - These are the model's confidence scores for each class. \n"
        f" - Higher values indicate a higher likelihood of the corresponding class. \n\n"
        f"**Uncertainty**:\n"
        f" - Non-Dementia: {uncertainty[0][0]:.2f}\n"
        f" - Dementia: {uncertainty[0][1]:.2f}\n\n"
        f"**What does this mean?**:\n"
        f" - This represents how certain the model is in its prediction. \n"
        f" - Lower uncertainty means more confidence in the prediction.",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10,
        transform=ax[1, 1].transAxes,
    )
    ax[1, 1].axis("off")

    output_path = os.path.join(output_dir, f"cdss_report_patient_{patient_id}.png")
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Report saved for {image_path} at {output_path}")


if __name__ == "__main__":
    # Load test filenames
    test_filenames = pd.read_csv("test_data/test_filenames.csv")
    random_test_filenames = test_filenames.sample(n=30, random_state=42).reset_index(
        drop=True
    )

    # Load the trained model
    model_path = "saved_model/VisionTransformerModel_2024_09_25-21_49_53.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming you have already defined the VisionTransformerModel class
    model = VisionTransformerModel(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Generate reports for the test set
    for i, row in random_test_filenames.iterrows():
        image_path = row["Filename"]
        output_dir = f"cdss_reports_lime2"

        generate_report(model, image_path, output_dir, device)
