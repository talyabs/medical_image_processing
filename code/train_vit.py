import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ViT_B_16_Weights
from train_cnn import *


class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformerModel, self).__init__()
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Freeze some layers
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze the last few transformer blocks
        for param in self.vit.encoder.layers[-4:].parameters():
            param.requires_grad = True

        # Modify the classification head
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.vit(x)


def get_transforms(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Convert to 3-channel grayscale
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def get_data_loaders(hparams):
    print(f"Loading dataset from {hparams.dataset_path}")
    dataset = datasets.ImageFolder(
        hparams.dataset_path, transform=transforms.Compose([transforms.Grayscale()])
    )

    hparams.num_classes = len(dataset.classes)
    print(f"Number of classes detected: {hparams.num_classes}")
    print(f"Classes: {dataset.classes}")

    test_split = 0.25
    val_split = hparams.val_split * (1 - test_split)

    train_idx, val_idx, test_idx = split_dataset_func(dataset, val_split, test_split)

    print(
        f"Dataset split sizes: Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}"
    )
    train_patients = set(get_patient_id(dataset.samples[i][0]) for i in train_idx)
    val_patients = set(get_patient_id(dataset.samples[i][0]) for i in val_idx)
    test_patients = set(get_patient_id(dataset.samples[i][0]) for i in test_idx)

    train_val_intersection = train_patients.intersection(val_patients)
    train_test_intersection = train_patients.intersection(test_patients)
    val_test_intersection = val_patients.intersection(test_patients)

    if train_val_intersection or train_test_intersection or val_test_intersection:
        print("WARNING: Patient overlap detected between splits!")
        print(f"Train-Val intersection: {train_val_intersection}")
        print(f"Train-Test intersection: {train_test_intersection}")
        print(f"Val-Test intersection: {val_test_intersection}")
    else:
        print("Patient uniqueness check passed: No overlap between splits.")

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    class_weights = get_class_weights(dataset)
    print(f"Class weights: {class_weights}")

    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(hparams.num_classes)], dtype=torch.float32
    )

    # Use class weights for sampling in the training set
    train_labels = [dataset.targets[i] for i in train_idx]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    data_transforms = get_transforms()

    train_dataset = CustomDataset(train_dataset, transform=data_transforms)
    val_dataset = CustomDataset(val_dataset, transform=data_transforms)
    test_dataset = CustomDataset(test_dataset, transform=data_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.train_batch_size,
        sampler=sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.train_batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=hparams.test_batch_size, shuffle=False
    )

    print(
        f"Data loaders created. Batch sizes - Train: {hparams.train_batch_size}, Test: {hparams.test_batch_size}"
    )

    for split_name, split_dataset in [
        ("Training", train_dataset),
        ("Validation", val_dataset),
        ("Test", test_dataset),
    ]:
        class_distribution = Counter(split_dataset.targets)
        print(f"\nClass distribution in {split_name} set:")
        for label, count in class_distribution.items():
            print(f"Class {label}: {count} samples")

    return train_loader, val_loader, test_loader, weight_tensor


def train_and_validate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    early_stopping,
):
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

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
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)

        val_loss, val_accuracy, f1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if early_stopping(val_loss):
            print(f"Early stopping at Epoch {epoch+1}")
            break

    return train_losses, train_accuracies, val_losses, val_accuracies


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "vit"

    hparams = Hparams()  # Assuming you have this class defined
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

    model = VisionTransformerModel(num_classes=hparams.num_classes).to(device)

    class_weights = class_weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams.learning_rate, weight_decay=0.05
    )

    early_stopping = EarlyStopping(patience=5, mode="min")

    train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        hparams.num_epochs,
        early_stopping,
    )

    print("Training and Validation completed.")
    print("Testing the model...")
    plot_losses(train_losses, val_losses, f"{model_name}_losses.png")
    plot_accuracies(train_accuracies, val_accuracies, f"{model_name}_accuracies.png")

    test_loss, test_accuracy, predictions, ground_truths = predict(
        model, test_loader, criterion, device, eval=True
    )

    save_model(model, hparams)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    class_to_idx = train_loader.dataset.dataset.class_to_idx

    # Generate and save the classification report
    save_classification_report(
        ground_truths,
        predictions,
        class_to_idx,
        filename=f"{model_name}_classification_report_{hparams.num_epochs}_epochs.txt",
    )

    plot_confusion_matrix(
        ground_truths,
        predictions,
        class_to_idx,
        filename=f"{model_name}_confusion_matrix_{hparams.num_epochs}_epochs.png",
    )
