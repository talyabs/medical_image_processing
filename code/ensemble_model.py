import torch
import torch.nn as nn
from models_utils import *
from train_cnn import CNN
from train_densenet import DenseNetModel
from train_inceptionv3 import InceptionV3Model
from train_resnet import ResNetModel


class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else torch.ones(len(models))
        self.weights = self.weights / self.weights.sum()

    def forward(self, x, mean=None, std=None):
        outputs = []
        for model in self.models:
            if isinstance(model, ResNetModel):
                # Convert grayscale to 3 channels for ResNet
                x_rgb = x.repeat(1, 3, 1, 1)
                out = model(x_rgb)
            elif isinstance(model, CNN):
                out = model(x, mean, std)
            else:
                out = model(x)
            outputs.append(out)
        weighted_outputs = [w * out for w, out in zip(self.weights, outputs)]
        return torch.stack(weighted_outputs).sum(dim=0)


def load_trained_model(model_class, model_path, device):
    model = model_class(num_classes=3).to(device)
    state_dict = torch.load(model_path)
    model_dict = model.state_dict()

    pretrained_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    model.eval()
    return model


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

            if isinstance(model, EnsembleModel):
                output = model(img, mean, std)
            else:
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


def main():
    hparams = Hparams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading trained models...")
    # Load trained models
    cnn_model = load_trained_model(
        CNN, "saved_model/CNN_2024_09_14-16_07_45.pt", device
    )
    resnet_model = load_trained_model(
        ResNetModel, "saved_model/ResNetWithExtras_2024_09_13-19_49_49.pt", device
    )
    densenet_model = load_trained_model(
        DenseNetModel, "saved_model/DenseNetModel_2024_09_14-18_42_26.pt", device
    )
    inception_model = load_trained_model(
        InceptionV3Model, "saved_model/InceptionV3Model_2024_09_15-12_43_19.pt", device
    )

    print("Creating ensemble model...")
    # Create ensemble model
    models = [cnn_model, resnet_model, densenet_model, inception_model]
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Equal weights, can be adjusted
    ensemble_model = EnsembleModel(models, weights).to(device)

    train_loader, val_loader, test_loader, weight_tensor = get_data_loaders(hparams)

    criterion = nn.CrossEntropyLoss()

    print("Testing the ensemble model...")
    test_loss, test_accuracy, predictions, ground_truths = predict(
        ensemble_model, test_loader, criterion, device, eval=True
    )

    print(
        f"Ensemble Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

    class_to_idx = train_loader.dataset.subset.dataset.class_to_idx
    save_classification_report(
        ground_truths,
        predictions,
        class_to_idx,
        filename="ensemble_classification_report.txt",
    )
    plot_confusion_matrix(
        ground_truths,
        predictions,
        class_to_idx,
        filename="ensemble_confusion_matrix.png",
    )

    save_model(ensemble_model, hparams)


if __name__ == "__main__":
    main()
