import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import numpy as np

from train_cnn import *


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
    def __init__(self, num_classes=2):  # Change num_classes default to 2
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
        return self.vit(x)

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
    print(f"Loading dataset from {hparams.dataset_path}")
    dataset = datasets.ImageFolder(hparams.dataset_path,
                                   transform=transforms.Compose([transforms.Grayscale()]))
    
    # Modify the dataset to combine dementia classes
    dementia_classes = ['Mild Dementia', 'Very mild Dementia', 'Moderate Dementia']
    dataset.samples = [(path, 1 if dataset.classes[label] in dementia_classes else 0) for path, label in dataset.samples]
    dataset.targets = [1 if dataset.classes[label] in dementia_classes else 0 for label in dataset.targets]
    dataset.class_to_idx = {'non_dementia': 0, 'dementia': 1}
    dataset.classes = ['non_dementia', 'dementia']
    
    hparams.num_classes = 2
    print(f"Number of classes: {hparams.num_classes}")
    print(f"Classes: {dataset.classes}")

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

    print(f"Dataset split sizes: Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Number of patients: Train: {len(train_patients)}, Test: {len(test_patients)}")

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    class_weights = get_class_weights(dataset)
    print(f"Class weights: {class_weights}")
    
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
    
    print(f"Data loaders created. Batch sizes - Train: {hparams.train_batch_size}, Test: {hparams.test_batch_size}")
    
    for split_name, split_dataset in [("Training", train_dataset), ("Test", test_dataset)]:
        class_distribution = Counter(split_dataset.targets)
        print(f"\nClass distribution in {split_name} set:")
        for label, count in class_distribution.items():
            print(f"Class {dataset.classes[label]}: {count} samples")
    
    return train_loader, test_loader, weight_tensor


def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses, train_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_idx, (data, _, _, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / len(train_loader.dataset)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = 'vit_binary'

    hparams = Hparams()
    hparams.num_classes = 2  # Ensure this is set to 2
    hparams.num_epochs = 5
    train_loader, test_loader, class_weights = get_data_loaders(hparams)

    model = VisionTransformerModel(num_classes=hparams.num_classes).to(device)
    
    class_weights = class_weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=0.05)

    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, device, hparams.num_epochs
    )
    
    print('Training completed.')
    print('Testing the model...')
    plot_losses(train_losses, filename=f'{model_name}_losses_{hparams.num_epochs}_ephocs.png')
    plot_accuracies(train_accuracies, filename=f'{model_name}_accuracies_{hparams.num_epochs}_ephocs.png')
    
    test_loss, test_accuracy, predictions, ground_truths = predict(model, test_loader, criterion, device, eval=True)
    
    save_model(model, hparams)
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    class_to_idx = train_loader.dataset.dataset.class_to_idx

    # Generate and save the classification report
    save_classification_report(ground_truths, predictions, class_to_idx, filename=f'{model_name}_classification_report_{hparams.num_epochs}_epochs.txt')
    
    plot_confusion_matrix(ground_truths, predictions, class_to_idx, filename=f'{model_name}_confusion_matrix_{hparams.num_epochs}_epochs.png')


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # model_name = 'vit_binary'

    # hparams = Hparams()
    # hparams.num_classes = 2  # Ensure this is set to 2
    # train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

    # model = VisionTransformerModel(num_classes=hparams.num_classes).to(device)
    
    # class_weights = class_weights.float().to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=0.05)
    
    # early_stopping = EarlyStopping(patience=5, mode='min')

    # train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(
    #     model, train_loader, val_loader, criterion, optimizer, device, hparams.num_epochs, early_stopping
    # )
    
    # print('Training and Validation completed.')
    # print('Testing the model...')
    # plot_losses(train_losses, val_losses, f'{model_name}_losses.png')
    # plot_accuracies(train_accuracies, val_accuracies, f'{model_name}_accuracies.png')
    
    # test_loss, test_accuracy, predictions, ground_truths = predict(model, test_loader, criterion, device, eval=True)
    
    # save_model(model, hparams)
    
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    # class_to_idx = train_loader.dataset.dataset.class_to_idx

    # # Generate and save the classification report
    # save_classification_report(ground_truths, predictions, class_to_idx, filename=f'{model_name}_classification_report_{hparams.num_epochs}_epochs.txt')
    
    # plot_confusion_matrix(ground_truths, predictions, class_to_idx, filename=f'{model_name}_confusion_matrix_{hparams.num_epochs}_epochs.png')
