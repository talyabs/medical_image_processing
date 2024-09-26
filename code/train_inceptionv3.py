import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import Inception_V3_Weights
from train_cnn import *


class CustomInceptionV3(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(CustomInceptionV3, self).__init__()

        original_model = models.inception_v3(weights=weights)

        self.Conv2d_1a_3x3 = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        original_weight = original_model.Conv2d_1a_3x3.conv.weight
        new_weight = original_weight.sum(dim=1, keepdim=True)
        self.Conv2d_1a_3x3.weight = nn.Parameter(new_weight)

        # Copy the rest of the layers
        self.Conv2d_2a_3x3 = original_model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = original_model.Conv2d_2b_3x3
        self.maxpool1 = original_model.maxpool1
        self.Conv2d_3b_1x1 = original_model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = original_model.Conv2d_4a_3x3
        self.maxpool2 = original_model.maxpool2

        self.Mixed_5b = original_model.Mixed_5b
        self.Mixed_5c = original_model.Mixed_5c

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(288, num_classes)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class InceptionV3Model(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3Model, self).__init__()
        self.inception = CustomInceptionV3(
            num_classes, weights=Inception_V3_Weights.IMAGENET1K_V1
        )

    def forward(self, x, mean=None, std=None):
        # Ignore mean and std inputs to match the existing train function
        return self.inception(x)


class InceptionLoss(nn.Module):
    def __init__(self, base_criterion):
        super(InceptionLoss, self).__init__()
        self.base_criterion = base_criterion

    def forward(self, output, target):
        return self.base_criterion(output, target)


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "inceptionv3"

    hparams = Hparams()
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

    model = InceptionV3Model(num_classes=hparams.num_classes).to(device)

    class_weights = class_weights.float().to(device)
    base_criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = InceptionLoss(base_criterion)
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
    plot_losses(train_losses, val_losses, f"{model_name}_losses.png")
    plot_accuracies(train_accuracies, val_accuracies, f"{model_name}_accuracies.png")

    test_loss, test_accuracy, predictions, ground_truths = predict(
        model, test_loader, criterion, device, eval=True
    )

    save_model(model, hparams)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    class_to_idx = train_loader.dataset.dataset.class_to_idx

    save_classification_report(
        ground_truths,
        predictions,
        class_to_idx,
        filename=f"{model_name}_classification_report_{hparams.num_epochs}_ephocs.txt",
    )

    plot_confusion_matrix(
        ground_truths,
        predictions,
        class_to_idx,
        filename=f"{model_name}_confusion_matrixt_{hparams.num_epochs}_ephocs.png",
    )
