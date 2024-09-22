import os
import torch
from train_vit_binary_with_validation_cdss import VisionTransformerModel, get_data_loaders
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from torchvision import datasets  # <-- Add this line

data_path = '/data/talya/medical_image_processing/Data'

class Hparams:
    def __init__(self, train_batch_size=64, test_batch_size=64, learning_rate=0.001, num_epochs=10, val_split=0.15, test_split=0.15, model_path='saved_model', dataset_path=data_path):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.val_split = val_split
        self.test_split = test_split
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.num_classes = None

def load_model(model, model_path):
    try:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_gradients(module, grad_input, grad_output):
    # Store gradients in the module itself
    module.saved_gradients = grad_output[0]
    


def create_cdss_reports(test_loader, model, device, dataset):
    print("Generating clinical decision support reports for test images...")
    
    num_reports = 10  # 5 reports where True class == Predicted class (for each class)
    target_folder = 'cdss_reports'
    
    # Create a folder to save the reports if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Set counters for true positive matches (True == Predicted)
    dementia_count = 0
    non_dementia_count = 0
    max_class_count = 5  # Limit to 5 images from each class

    # Class name mapping
    class_names = {0: 'Non-Dementia', 1: 'Dementia'}

    test_iter = iter(test_loader)

    while dementia_count < max_class_count or non_dementia_count < max_class_count:
        # Adjust to unpack four values: (image, mean, std, label)
        data, mean, std, label = next(test_iter)
        data, label = data.to(device), label.to(device)

        # Forward pass and get classification score for each image in the batch
        with torch.no_grad():
            output = model(data)
            softmax_output = F.softmax(output, dim=1)
        
        # Process each image in the batch
        for j in range(data.size(0)):  # Iterate over the batch size
            score = float(softmax_output[j].max().item())  # Ensure score is a float
            pred_class = softmax_output[j].argmax().item()
            true_class = label[j].item()  # Get the true class

            # Ensure 5 examples where True class == Predicted class
            if true_class == 1 and pred_class == 1 and dementia_count < max_class_count:
                dementia_count += 1
            elif true_class == 0 and pred_class == 0 and non_dementia_count < max_class_count:
                non_dementia_count += 1
            else:
                continue  # Skip images that don't meet the conditions

            # Get the original file path from the dataset
            image_path = dataset.samples[j][0]

            # Generate attention map for the prediction
            attention_map = generate_attention_map(model, data[j], pred_class)

            # Resize the original image and attention map for better visibility (bigger size)
            original_image = transforms.ToPILImage()(data[j].cpu()).resize((500, 500))  # Resize the image
            attention_map = attention_map.resize((500, 500))  # Resize the attention map
            
            # Map true class and predicted class to class names
            true_class_name = class_names[true_class]
            pred_class_name = class_names[pred_class]

            # Save report to dedicated folder
            report_filename = os.path.join(target_folder, f'report_{dementia_count + non_dementia_count}_image_{j+1}.png')
            generate_report(original_image, attention_map, true_class_name, pred_class_name, score, image_path, report_filename)

    print(f'{num_reports} reports generated in {target_folder}.')

def generate_report(image, attention_map, true_class_name, pred_class_name, score, image_path, filename='report.png'):
    # Create a larger blank image for the report
    report_image = Image.new('RGB', (900, 600), color='white')  # Adjusting the width for more space
    draw = ImageDraw.Draw(report_image)
    
    # Load a default font
    font = ImageFont.load_default()

    # Place the original image and attention map side by side
    report_image.paste(image, (50, 100))
    
    # If the attention map exists and is valid, paste it. Otherwise, skip it.
    if attention_map:
        report_image.paste(attention_map, (500, 100))  # Adjusted positioning
    

    # Add the image path at the top of the report
    draw.text((50, 10), f"Image Path: {image_path}", font=font, fill="black")  # File path
    # Add title with true class and predicted class
    title = f"True Class: {true_class_name}, Predicted Class: {pred_class_name}"
    draw.text((50, 30), title, font=font, fill="black")  # Title on top
    
    # Add the confidence score directly under the title
    confidence_text = f"Confidence: {score:.2f}"  # Rounded to 2 decimal places
    draw.text((50, 60), confidence_text, font=font, fill="black")  # Placed below the title
    
    # Save the report
    report_image.save(filename)
    print(f"Report saved to {filename}")




def generate_attention_map(model, input_image, target_class):
    model.eval()
    
    # Dictionary to store gradients and activations
    gradients = {}
    activations = {}

    def save_grad(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]
    
    def save_activation(module, input, output):
        activations["value"] = output

    # Register hooks to save gradients and activations
    grad_handle = model.vit.encoder.layers[-1].register_backward_hook(save_grad)
    activation_handle = model.vit.encoder.layers[-1].register_forward_hook(save_activation)
    
    # Forward pass
    output = model(input_image.unsqueeze(0))
    pred_class = output.argmax(dim=1).item()

    # Create target tensor on the same device as output
    target = torch.zeros_like(output).to(output.device)
    target[0, target_class] = 1

    # Backward pass to calculate gradients
    output.backward(gradient=target)
    
    # Remove hooks after backward pass
    grad_handle.remove()
    activation_handle.remove()
    
    # Retrieve gradients and activations from the dictionary
    grad_value = gradients["value"]
    activation_value = activations["value"]

    # Remove the CLS token (first token)
    grad_value = grad_value[:, 1:]  # Exclude the first token
    activation_value = activation_value[:, 1:]  # Exclude the first token

    # Compute the weighted sum of the gradients for each patch
    weights = torch.mean(grad_value, dim=1, keepdim=True)  # Mean over sequence length (patches)
    cam = torch.sum(weights * activation_value, dim=2).squeeze().detach()  # Sum over the hidden dimension

    # Reshape CAM to match the patch grid (e.g., 14x14 patches for ViT-B/16)
    num_patches = int(cam.shape[0])  # Should be a square number, e.g., 196 (14x14)
    patch_size = int(num_patches ** 0.5)
    cam = cam.view(patch_size, patch_size)  # Reshape to 2D grid

    # Normalize the attention map
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Convert to PIL image (now that it is 2D)
    heatmap = ToPILImage()(cam)

    return heatmap



if __name__ == "__main__":
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = 'vit_binary'
    model_path = '/data/talya/medical_image_processing/saved_model/VisionTransformerModel_2024_09_21-19_14_32.pt'

    hparams = Hparams()
    hparams.num_classes = 2  # Ensure this is set to 2
    hparams.num_epochs = 2
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(hparams)

    # Initialize the model
    model = VisionTransformerModel(num_classes=hparams.num_classes).to(device)

    # Load the specific saved model
    model = load_model(model, model_path)
   
   

    if model is not None:
        # Now that the model is loaded, generate the CDSS reports
        dataset = datasets.ImageFolder(hparams.dataset_path,
                                   transform=transforms.Compose([transforms.Grayscale()]))
        create_cdss_reports(test_loader, model, device, dataset)
    else:
        print("Model loading failed. Exiting.")
