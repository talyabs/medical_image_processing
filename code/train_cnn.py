import torch
import torch.nn as nn
import torch.optim as optim
from models_utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

data_path = "/data/talya/medical_image_processing/Data"
num_epochs = 10


class ConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.pool(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.fc = nn.Linear(in_channel, out_channel)
        self.batch_norm = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class CNN(nn.Module):
    def __init__(self, num_classes=3):  # Default to 3 classes, but can be changed
        super().__init__()

        self.convblock1 = ConvBlock(1, 32, 64)
        self.convblock2 = ConvBlock(64, 128, 128)
        self.convblock3 = ConvBlock(128, 256, 256)
        self.convblock4 = ConvBlock(256, 512, 512)

        self.flatten = nn.Flatten(start_dim=1)

        self.linearblock1 = LinearBlock(512 * 15 * 15, 1024)
        self.linearblock2 = LinearBlock(1024, 512)
        self.linearblock3 = LinearBlock(512, 16)

        self.linearblock4 = LinearBlock(16 + 2, num_classes)  # Changed this line
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img, mean, std):
        x = self.convblock1(img)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)

        x = self.flatten(x)

        x = self.linearblock1(x)
        x = self.linearblock2(x)
        x = self.linearblock3(x)

        x = torch.concat([x, mean.unsqueeze(1), std.unsqueeze(1)], dim=-1)
        x = self.linearblock4(x)
        x = self.softmax(x)

        return x


def cnn_model():
    hparams = Hparams()
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

    model = CNN(num_classes=hparams.num_classes).to(device)

    class_weights = class_weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5, verbose=True
    )

    early_stopping = EarlyStopping(patience=3, mode="max")

    train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        hparams.num_epochs,
        early_stopping,
    )

    print("Training and Validation completed.")
    print("Testing the model...")
    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accuracies, val_accuracies)

    test_loss, test_accuracy, predictions, ground_truths = predict(
        model, test_loader, criterion, device, eval=True
    )

    save_model(model, hparams)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    class_to_idx = train_loader.dataset.subset.dataset.class_to_idx

    # Generate and save the classification report
    save_classification_report(
        ground_truths, predictions, class_to_idx, filename="classification_report.txt"
    )

    plot_confusion_matrix(
        ground_truths, predictions, class_to_idx, filename="confusion_matrix.png"
    )


if __name__ == "__main__":
    cnn_model()


# Number of files in directory 'Data/Very mild Dementia': 13725
# Number of files in directory 'Data/Non Demented': 67222
# Number of files in directory 'Data/Mild Dementia': 5002
