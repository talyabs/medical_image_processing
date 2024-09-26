import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import DenseNet121_Weights
from train_cnn import *


class DenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.densenet(x)


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hparams = Hparams()
    hparams.num_epochs = 8
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

    model = DenseNetModel(num_classes=hparams.num_classes).to(device)

    class_weights = class_weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)

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
    plot_losses(train_losses, val_losses, "desndet_losses.png")
    plot_accuracies(train_accuracies, val_accuracies, "densenet_accuracies.png")

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
        filename=f"densenet_classification_report_{hparams.num_epochs}_ephocs.txt",
    )

    plot_confusion_matrix(
        ground_truths,
        predictions,
        class_to_idx,
        filename=f"densenet_confusion_matrixt_{hparams.num_epochs}_ephocs.png",
    )
