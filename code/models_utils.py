import os
import re
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

data_path = "/data/talya/medical_image_processing/Data"
num_epochs = 30


class Hparams:
    def __init__(
        self,
        train_batch_size=64,
        test_batch_size=64,
        learning_rate=0.001,
        num_epochs=num_epochs,
        val_split=0.15,
        test_split=0.15,
        model_path="saved_model",
        dataset_path=data_path,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.val_split = val_split
        self.test_split = test_split
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.num_classes = None


class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.counter = 0
        self.patience = patience
        self.early_stop = False
        self.mode = mode

        if self.mode == "max":
            self.ref_value = float("-inf")
        elif self.mode == "min":
            self.ref_value = float("inf")
        else:
            raise Exception(
                f"Undefined mode for EarlyStopping - mode: {mode}\n"
                'Available modes are ["max", "min"]'
            )

    def __call__(self, value):
        if self.mode == "max":
            if value <= self.ref_value:
                self.counter += 1
            else:
                self.counter = 0
                self.ref_value = value
        elif self.mode == "min":
            if value >= self.ref_value:
                self.counter += 1
            else:
                self.counter = 0
                self.ref_value = value

        if self.counter == self.patience:
            self.early_stop = True


class CustomDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.indices = subset.indices
        self.dataset = subset.dataset

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        mean = torch.mean(x)
        std = torch.std(x)
        return x, mean, std, y

    def __len__(self):
        return len(self.subset)

    @property
    def targets(self):
        return [self.dataset.targets[i] for i in self.indices]


def get_transforms(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]
        )


def get_info_from_filename(filepath):
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


def get_sample_weights(dataset, train_dataset):
    y_train_indices = train_dataset.indices
    y_train = [dataset.targets[i] for i in y_train_indices]

    class_sample_counts = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
    )

    weights = 1.0 / class_sample_counts
    sample_weights = np.array([weights[t] for t in y_train])
    sample_weights = torch.from_numpy(sample_weights)

    return sample_weights


def get_class_weights(dataset):
    labels = [sample[1] for sample in dataset.samples]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    return dict(enumerate(class_weights))


def get_patient_id(filepath):
    filename = os.path.basename(filepath)
    match = re.match(r"OAS1_(\d+)_MR\d+_mpr-\d+_\d+.jpg", filename)
    return int(match.group(1)) if match else None


def split_dataset_func(dataset, val_split, test_split):
    patient_class_groups = defaultdict(lambda: defaultdict(list))
    for idx, (filepath, label) in enumerate(dataset.samples):
        patient_id = get_patient_id(filepath)
        if patient_id is not None:
            patient_class_groups[label][patient_id].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for label, patients in patient_class_groups.items():
        patient_ids = sorted(patients.keys())
        n_patients = len(patient_ids)

        n_test = int(n_patients * test_split)
        n_val = int(n_patients * val_split)
        n_train = n_patients - n_test - n_val

        # Ensure at least one patient in each split
        n_test = max(n_test, 1)
        n_val = max(n_val, 1)
        n_train = max(n_train, 1)

        # Adjust if necessary
        while n_train + n_val + n_test > n_patients:
            if n_train > n_val and n_train > n_test:
                n_train -= 1
            elif n_val > n_test:
                n_val -= 1
            else:
                n_test -= 1

        # Split patient IDs
        test_patients = patient_ids[:n_test]
        val_patients = patient_ids[n_test : n_test + n_val]
        train_patients = patient_ids[n_test + n_val :]

        # Add indices to respective splits
        for patient in test_patients:
            test_idx.extend(patients[patient])
        for patient in val_patients:
            val_idx.extend(patients[patient])
        for patient in train_patients:
            train_idx.extend(patients[patient])

    return train_idx, val_idx, test_idx


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

    train_transform = get_transforms(train=True)
    val_test_transform = get_transforms(train=False)

    train_dataset = CustomDataset(train_dataset, transform=train_transform)
    val_dataset = CustomDataset(val_dataset, transform=val_test_transform)
    test_dataset = CustomDataset(test_dataset, transform=val_test_transform)

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

    # Print class distribution for each split
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


def print_class_distribution(dataset, split_name):
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    total_samples = len(labels)

    print(f"\nClass distribution for {split_name} set:")
    for class_label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")


def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    batch_size = 0

    targets, preds = [], []

    for batch_idx, (img, mean, std, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)
        mean, std = mean.to(device), std.to(device)
        batch_size = len(img)

        optimizer.zero_grad()

        output = model(img)

        loss = criterion(output, target)

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)

        targets.extend(target.cpu().numpy())
        preds.extend(pred.cpu().numpy().flatten())

        train_correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
            )

    train_length = len(train_loader.dataset)
    train_loss /= train_length
    train_accuracy = 100.0 * train_correct / train_length
    f1 = f1_score(targets, preds, average="macro")

    return train_loss, train_accuracy, f1


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0

    targets, preds = [], []

    with torch.no_grad():
        for img, mean, std, target in val_loader:
            img, target = img.to(device), target.to(device)
            mean, std = mean.to(device), std.to(device)  # Keep these in case needed

            try:
                output = model(img, mean, std)
            except TypeError:
                output = model(img)

            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)

            targets.extend(target.cpu().numpy())
            preds.extend(pred.cpu().numpy().flatten())

            val_correct += pred.eq(target.view_as(pred)).sum().item()

    val_length = len(val_loader.dataset)
    val_loss /= val_length
    val_accuracy = 100.0 * val_correct / val_length
    f1 = f1_score(targets, preds, average="macro")

    return val_loss, val_accuracy, f1


def predict(model, data_loader, criterion, device, eval=False):
    model.eval()
    pred_loss = 0
    pred_correct = 0
    total_size = 0

    predictions = torch.IntTensor()
    ground_truths = torch.IntTensor()

    predictions, ground_truths = predictions.to(device), ground_truths.to(device)

    with torch.no_grad():
        for batch_idx, (img, mean, std, target) in enumerate(data_loader):
            img, target = img.to(device), target.to(device)
            mean, std = mean.to(device), std.to(device)  # Keep these in case needed

            # Try to use all inputs, if it fails, use only the image
            try:
                output = model(img, mean, std)
            except TypeError:
                output = model(img)

            loss = criterion(output, target)
            pred_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            pred_correct += pred.eq(target.view_as(pred)).sum().item()

            predictions = torch.cat((predictions, pred), dim=0)
            ground_truths = torch.cat((ground_truths, target), dim=0)

            total_size += len(img)

    pred_loss /= total_size
    pred_accuracy = 100.0 * pred_correct / total_size

    if eval:
        return (
            pred_loss,
            pred_accuracy,
            predictions.cpu().numpy(),
            ground_truths.cpu().numpy(),
        )
    else:
        return predictions.cpu().numpy(), ground_truths.cpu().numpy()


def train_and_validate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    early_stopping=None,
):
    train_losses, train_accuracies, train_f1_scores = [], [], []
    val_losses, val_accuracies, val_f1_scores = [], [], []

    for epoch in range(num_epochs):
        # Training
        train_loss, train_accuracy, train_f1 = train(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)

        # Validation
        val_loss, val_accuracy, val_f1 = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train F1: {train_f1:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}"
        )

        scheduler.step(val_accuracy)

        if early_stopping is not None:
            early_stopping(val_accuracy)
            if early_stopping.early_stop:
                print(f"Early stopping at Epoch {epoch+1}")
                break

    return train_losses, train_accuracies, val_losses, val_accuracies


def plot_losses(train_losses, val_losses, filename="losses_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Save the plot to a file
    filename = "plots/" + filename
    plt.savefig(filename)
    plt.close()


def plot_accuracies(train_accuracies, val_accuracies, filename="accuracies_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    # Save the plot to a file
    filename = "plots/" + filename
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(
    y_true, y_pred, class_to_idx, filename="confusion_matrix.png"
):
    conf_mat = confusion_matrix(y_true, y_pred)
    class_names = list(class_to_idx.keys())
    df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")

    # Save the plot to a file
    filename = "plots/" + filename
    plt.savefig(filename)
    plt.close()


def save_classification_report(
    y_true, y_pred, class_to_idx, filename="classification_report.txt"
):
    # Generate the classification report
    class_names = list(class_to_idx.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Write the report to a text file
    filename = "plots/" + filename
    with open(filename, "w") as file:
        file.write(report)


def save_model(model, hparams):
    os.makedirs(hparams.model_path, exist_ok=True)

    model_name = (
        model.__class__.__name__
        + "_"
        + datetime.now().strftime("%Y_%m_%d-%H_%M_%S" + ".pt")
    )

    try:
        torch.save(model.state_dict(), os.path.join(hparams.model_path, model_name))
        return True
    except:
        return False
