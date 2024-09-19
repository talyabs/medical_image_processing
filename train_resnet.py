import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

import numpy as np
from torchvision.models import ResNet50_Weights


from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from train_cnn import *



def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

def get_class_weights(dataset):
    labels = [sample[1] for sample in dataset.samples]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(class_weights))

from collections import Counter

def get_data_loaders(hparams):
    print(f"Loading dataset from {hparams.dataset_path}")
    dataset = datasets.ImageFolder(hparams.dataset_path, transform=get_transforms())
    
    hparams.num_classes = len(dataset.classes)
    print(f"Number of classes detected: {hparams.num_classes}")
    print(f"Classes: {dataset.classes}")
    
    test_split = 0.25
    val_split = hparams.val_split * (1 - test_split)
    
    train_idx, val_idx, test_idx = split_dataset_func(dataset, val_split, test_split)
    
    print(f"Dataset split sizes: Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    class_weights = get_class_weights(dataset)
    print(f"Class weights: {class_weights}")
    
    weight_tensor = torch.tensor([class_weights[i] for i in range(hparams.num_classes)], dtype=torch.float32)
    
    train_labels = [dataset.targets[i] for i in train_idx]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=hparams.train_batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams.train_batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams.test_batch_size, shuffle=False)
    
    print(f"Data loaders created. Batch sizes - Train: {hparams.train_batch_size}, Test: {hparams.test_batch_size}")
    
    # Print class distribution for each split
    for split_name, split_dataset, split_idx in [
        ("Training", train_dataset, train_idx),
        ("Validation", val_dataset, val_idx),
        ("Test", test_dataset, test_idx)
    ]:
        split_targets = [dataset.targets[i] for i in split_idx]
        class_distribution = Counter(split_targets)
        print(f"\nClass distribution in {split_name} set:")
        for label, count in class_distribution.items():
            print(f"Class {label}: {count} samples")
    
    return train_loader, val_loader, test_loader, weight_tensor


class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        # Use the new weights parameter
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')




# ... (keep the previous imports and utility functions)

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation
        val_loss, val_acc = predict(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def predict(model, data_loader, criterion, device, eval=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total
    
    if eval:
        return avg_loss, accuracy, all_predictions, all_labels
    else:
        return avg_loss, accuracy

def save_model(model, hparams):
    torch.save(model.state_dict(), f"{hparams.model_path}/resnet_model.pth")
    print(f"Model saved to {hparams.model_path}/resnet_model.pth")

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig('plots/loss_plot.png')
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracies')
    plt.savefig('plots/accuracy_plot.png')
    plt.close()

def save_classification_report(y_true, y_pred, class_to_idx, filename='classification_report.txt'):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    filename = 'plots/' + filename
    with open(filename, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {filename}")

def plot_confusion_matrix(y_true, y_pred, class_to_idx, filename='confusion_matrix.png'):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    filename = 'plots/' + filename
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hparams = Hparams()  # Assuming you have this class defined
    hparams.num_epochs = 8

    train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

    model = ResNetModel(num_classes=hparams.num_classes).to(device)
    
    class_weights = class_weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
    
    early_stopping = EarlyStopping(patience=5, mode='min')

    train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, device, hparams.num_epochs, early_stopping
    )
    
    print('Training and Validation completed.')
    print('Testing the model...')
    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accuracies, val_accuracies)
    
    test_loss, test_accuracy, predictions, ground_truths = predict(model, test_loader, criterion, device, eval=True)
    
    save_model(model, hparams)
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    class_to_idx = train_loader.dataset.dataset.class_to_idx

    # Generate and save the classification report
    save_classification_report(ground_truths, predictions, class_to_idx, filename=f'classification_report_resnet_{hparams.num_epochs}_ephocs.txt')
    
    plot_confusion_matrix(ground_truths, predictions, class_to_idx, filename=f'confusion_matrix_resnet_{hparams.num_epochs}_ephocs.png')

# # Main execution
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     hparams = Hparams()  # Assuming you have this class defined
#     train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

#     model = ResNetModel(num_classes=hparams.num_classes).to(device)
    
#     class_weights = class_weights.float().to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)

#     train_model(model, train_loader, val_loader, criterion, optimizer, hparams.num_epochs, device)

#     # Evaluate on test set
#     model.eval()
#     test_correct = 0
#     test_total = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             test_total += labels.size(0)
#             test_correct += predicted.eq(labels).sum().item()
    
#     test_acc = test_correct / test_total
#     print(f'Test Accuracy: {test_acc:.4f}')