
import torch
import torch.nn as nn

import torch.optim as optim
import torchvision.models as models

from torch.optim.lr_scheduler import OneCycleLR

from  models_utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device.")

data_path = '/data/talya/medical_image_processing/Data'
num_epochs = 10

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        
    def forward(self, x):
        # First convolution
        attention = self.conv1(x)
        attention = F.relu(attention)
        
        # Second convolution
        attention = self.conv2(attention)
        
        # Apply sigmoid to get attention weights
        attention = torch.sigmoid(attention)
        
        # Apply attention weights to input feature map
        out = x * attention
        
        return out

class AttentionCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(AttentionCNN, self).__init__()
        
        # Load a pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Add spatial attention layer
        self.attention = SpatialAttention(2048)  # 2048 is the number of channels in the last ResNet layer
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)


class AttentionCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(AttentionCNN, self).__init__()
        
        # Load a pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Add spatial attention layer
        self.attention = SpatialAttention(2048)  # 2048 is the number of channels in the last ResNet layer
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # ResNet features
        x = self.resnet(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



def cnn_model():
    hparams = Hparams()
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

    model = AttentionCNN(num_classes=hparams.num_classes).to(device)
    class_weights = class_weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    #optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=1e-5)

    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=hparams.num_epochs, steps_per_epoch=len(train_loader))

    
    early_stopping = EarlyStopping(patience=3, mode='max')

    train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, hparams.num_epochs, early_stopping)
    
    print('Training and Validation completed.')
    print('Testing the model...')
    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accuracies, val_accuracies)
    
    test_loss, test_accuracy, predictions, ground_truths = predict(model, test_loader, criterion, device, eval=True)
    
    save_model(model, hparams)
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    class_to_idx = train_loader.dataset.subset.dataset.class_to_idx

    # Generate and save the classification report
    save_classification_report(ground_truths, predictions, class_to_idx, filename='classification_report.txt')
    
    plot_confusion_matrix(ground_truths, predictions, class_to_idx, filename='confusion_matrix.png')

    
if __name__ == '__main__':
    cnn_model()
# Number of files in directory 'Data/Very mild Dementia': 13725
# Number of files in directory 'Data/Non Demented': 67222
# Number of files in directory 'Data/Mild Dementia': 5002
