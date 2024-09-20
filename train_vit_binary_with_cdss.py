import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import logging
from PIL import Image
import matplotlib.pyplot as plt

from train_cnn import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class Hparams:
    def __init__(self, train_batch_size=64, test_batch_size=64, learning_rate=0.001, num_epochs=num_epochs, val_split=0.15, test_split=0.15, model_path='saved_model', dataset_path=data_path):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.val_split = val_split
        self.test_split = test_split
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.num_classes = None


class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(VisionTransformerModel, self).__init__()
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Freeze some layers
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few transformer blocks
        for param in self.vit.encoder.layers[-4:].parameters():
            param.requires_grad = True
        
        # Modify the classification head for binary classification
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        logits = self.vit(x)
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities
    

class CustomGradCAM(GradCAM):
    def forward(self, input_tensor, targets=None, eigen_smooth=False):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = outputs.argmax(dim=-1)
            targets = [target_categories]
        else:
            targets = [targets]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target * output for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of our cases, we'll just need the features/activations computed during the forward pass.
        with torch.no_grad():
            activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
            grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        cam = self.compute_cam_per_layer(activations, grads, targets)
        return self.aggregate_multi_layers(cam)

def apply_grad_cam(model, image, target_category=None):
    target_layer = model.vit.encoder.layers[-1]
    
    cam = CustomGradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    
    model.eval()  # Ensure the model is in eval mode
    
    # Enable gradients temporarily
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=target_category)
    
    return grayscale_cam[0, :]
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_data_loaders(hparams):
    logging.info(f"Loading dataset from {hparams.dataset_path}")
    dataset = datasets.ImageFolder(hparams.dataset_path,
                                   transform=transforms.Compose([transforms.Grayscale()]))
    
    # Modify the dataset to combine dementia classes
    dementia_classes = ['Mild Dementia', 'Very mild Dementia', 'Moderate Dementia']
    dataset.samples = [(path, 1 if dataset.classes[label] in dementia_classes else 0) for path, label in dataset.samples]
    dataset.targets = [1 if dataset.classes[label] in dementia_classes else 0 for label in dataset.targets]
    dataset.class_to_idx = {'non_dementia': 0, 'dementia': 1}
    dataset.classes = ['non_dementia', 'dementia']
    
    hparams.num_classes = 2
    logging.info(f"Number of classes: {hparams.num_classes}")
    logging.info(f"Classes: {dataset.classes}")

    # Group images by patient
    patient_dict = {}
    for idx, (path, label) in enumerate(dataset.samples):
        patient_id = get_patient_id(path)
        if patient_id not in patient_dict:
            patient_dict[patient_id] = []
        patient_dict[patient_id].append((idx, label))

    # Split patients into train and test sets
    patient_ids = list(patient_dict.keys())
    patient_labels = [patient_dict[pid][0][1] for pid in patient_ids]  # Use the first image's label for each patient
    train_patients, test_patients = train_test_split(patient_ids, test_size=0.25, stratify=patient_labels)

    # Create train and test indices
    train_idx = [idx for pid in train_patients for idx, _ in patient_dict[pid]]
    test_idx = [idx for pid in test_patients for idx, _ in patient_dict[pid]]

    logging.info(f"Dataset split sizes: Train: {len(train_idx)}, Test: {len(test_idx)}")
    logging.info(f"Number of patients: Train: {len(train_patients)}, Test: {len(test_patients)}")

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    class_weights = get_class_weights(dataset)
    logging.info(f"Class weights: {class_weights}")
    
    weight_tensor = torch.tensor([class_weights[i] for i in range(hparams.num_classes)], dtype=torch.float32)
    
    train_labels = [dataset.targets[i] for i in train_idx]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    
    train_dataset = CustomDataset(train_dataset, transform=train_transform)
    test_dataset = CustomDataset(test_dataset, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=hparams.train_batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams.test_batch_size, shuffle=False)
    
    logging.info(f"Data loaders created. Batch sizes - Train: {hparams.train_batch_size}, Test: {hparams.test_batch_size}")
    
    for split_name, split_dataset in [("Training", train_dataset), ("Test", test_dataset)]:
        class_distribution = Counter(split_dataset.targets)
        logging.info(f"\nClass distribution in {split_name} set:")
        for label, count in class_distribution.items():
            logging.info(f"Class {dataset.classes[label]}: {count} samples")
    
    return train_loader, test_loader, weight_tensor


def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses, train_accuracies = [], []

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_idx, (data, _, _, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, probabilities = model(data)
            loss = criterion(logits, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = probabilities.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / len(train_loader.dataset)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    return train_losses, train_accuracies


def plot_losses(train_losses, val_losses=None, filename='losses_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss' if val_losses is None else 'Training and Validation Loss')
    
    # Save the plot to a file
    filename = 'plots/' + filename
    plt.savefig(filename)
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies=None, filename='accuracies_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    if val_accuracies is not None:
        plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy' if val_accuracies is None else 'Training and Validation Accuracy')
    
    # Save the plot to a file
    filename = 'plots/' + filename
    plt.savefig(filename)
    plt.close()




def visualize_grad_cam(image, cam, prediction, probability, save_path=None):
    plt.figure(figsize=(12, 5))
    
    class_names = ['Non-Dementia', 'Dementia']
    pred_class = class_names[prediction]
    
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f'Original Image\nPrediction: {pred_class} ({probability:.2f})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f'Grad-CAM\nPrediction: {pred_class} ({probability:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()



def generate_clinical_report(prediction, probabilities, cam_intensity, original_image, grad_cam, save_path):
    class_names = ['Non-Dementia', 'Dementia']
    report = f"Prediction: {class_names[prediction]}\n"
    report += f"Confidence: {probabilities[prediction]:.2f}\n"
    report += f"Probability of Non-Dementia: {probabilities[0]:.2f}\n"
    report += f"Probability of Dementia: {probabilities[1]:.2f}\n"
    
    # Analyze CAM intensity
    cam_mean = np.mean(cam_intensity)
    cam_max = np.max(cam_intensity)
    report += f"Average attention intensity: {cam_mean:.2f}\n"
    report += f"Peak attention intensity: {cam_max:.2f}\n"
    
    if cam_mean > 0.5:
        report += "The model shows strong activation across large areas of the brain.\n"
    elif cam_mean > 0.3:
        report += "The model shows moderate activation in several areas of the brain.\n"
    else:
        report += "The model shows relatively weak or localized activation patterns.\n"
    
    # Generate and save the image
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f'Original Image\nPrediction: {class_names[prediction]} ({probabilities[prediction]:.2f})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(original_image.squeeze().cpu().numpy(), cmap='gray')
    plt.imshow(grad_cam, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    report += f"\nGrad-CAM visualization has been saved to: {save_path}\n"
    report += "Please refer to the Grad-CAM visualization for specific regions of interest."
    
    return report


def predict(model, data_loader, criterion, device, eval=False, generate_cam=False):
    model.eval()
    pred_loss = 0
    pred_correct = 0
    total_size = 0

    predictions = []
    ground_truths = []
    grad_cams = []

    with torch.no_grad():
        for batch_idx, (data, _, _, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            logits, probabilities = model(data)
            loss = criterion(logits, target)
            pred_loss += loss.item()
            pred = probabilities.argmax(dim=1, keepdim=True)
            pred_correct += pred.eq(target.view_as(pred)).sum().item()

            predictions.extend(pred.cpu().numpy())
            ground_truths.extend(target.cpu().numpy())
            
            if generate_cam:
                for i in range(data.size(0)):
                    cam = apply_grad_cam(model, data[i].to(device))
                    grad_cams.append(cam)
            
            total_size += len(data)
    
    pred_loss /= total_size
    pred_accuracy = 100. * pred_correct / total_size

    if eval:
        return pred_loss, pred_accuracy, predictions, ground_truths, grad_cams if generate_cam else None
    else:
        return predictions, ground_truths, grad_cams if generate_cam else None
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model_name = 'vit_binary'

    hparams = Hparams()
    hparams.num_classes = 2  # Ensure this is set to 2
    hparams.num_epochs = 1
    train_loader, test_loader, class_weights = get_data_loaders(hparams)

    model = VisionTransformerModel(num_classes=hparams.num_classes).to(device)
    
    class_weights = class_weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=0.05)

    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, device, hparams.num_epochs
    )
    
    logging.info('Training completed.')
    logging.info('Testing the model...')
    plot_losses(train_losses, filename=f'{model_name}_losses_{hparams.num_epochs}_epochs.png')
    plot_accuracies(train_accuracies, filename=f'{model_name}_accuracies_{hparams.num_epochs}_epochs.png')
    
    test_loss, test_accuracy, predictions, ground_truths, grad_cams = predict(model, test_loader, criterion, device, eval=True, generate_cam=True)
    
    save_model(model, hparams)
    
    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    class_to_idx = train_loader.dataset.dataset.class_to_idx

    # Generate and save the classification report
    save_classification_report(ground_truths, predictions, class_to_idx, filename=f'{model_name}_classification_report_{hparams.num_epochs}_epochs.txt')
    
    plot_confusion_matrix(ground_truths, predictions, class_to_idx, filename=f'{model_name}_confusion_matrix_{hparams.num_epochs}_epochs.png')

    
   # In your main execution block:
    num_samples = min(5, len(grad_cams))  # Generate reports for up to 5 samples
    for i in range(num_samples):
        image, _, _, _ = test_loader.dataset[i]
        image = image.to(device)
        
        with torch.no_grad():
            logits, probabilities = model(image.unsqueeze(0))
        
        prediction = probabilities.argmax(dim=1).item()
        grad_cam_path = f'grad_cam_sample_{i}.png'
        
        report = generate_clinical_report(
            prediction, 
            probabilities[0].cpu().numpy(), 
            grad_cams[i],
            image,
            grad_cams[i],
            grad_cam_path
        )
        
        logging.info(f"Report for sample {i}:")
        logging.info(report)
        logging.info("-----------------------")

        # Optionally, save the report to a text file
        with open(f'clinical_report_sample_{i}.txt', 'w') as f:
            f.write(report)