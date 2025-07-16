import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

from modeling.ifrnet import IFRNet
from modeling.vgg import VGG16FeatLayer

def load_image(image_path, img_size=256):
    """Load and preprocess the input image."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")

def save_image(tensor, output_path):
    """Save the output tensor as an image."""
    transform = transforms.ToPILImage()
    try:
        image = tensor.squeeze(0).clamp(0, 1)  # Remove batch dimension and clamp values
        image = transform(image)
        image.save(output_path)
        print(f"Unfiltered image saved to {output_path}")
    except Exception as e:
        raise ValueError(f"Error saving image to {output_path}: {str(e)}")

def remove_filter(input_image_path, output_image_path, model_path='weights/ifrnet.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Remove Instagram filter from an image using the pretrained IFRNet model."""
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. Ensure ifrnet.pth is in the weights folder."
        )

    # Initialize VGG16 for feature extraction
    try:
        from torchvision.models import VGG16_Weights
        vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval().to(device)
        vgg_feat_layer = VGG16FeatLayer(vgg16, device)
    except Exception as e:
        raise RuntimeError(f"Error initializing VGG16 model: {str(e)}")

    # Initialize IFRNet model
    try:
        model = IFRNet(base_n_channels=32, destyler_n_channels=32)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if this is a training checkpoint with multiple models
        if "ifr" in checkpoint:
            # Extract just the IFRNet model weights
            model.load_state_dict(checkpoint["ifr"])
            print(f"Successfully loaded IFRNet weights from training checkpoint")
        else:
            # Try direct loading
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(
            f"Error loading model from {model_path}: {str(e)}. Ensure ifrnet.pth is compatible with the IFRNet architecture."
        )

    # Load and preprocess the image
    image = load_image(input_image_path)
    image = image.to(device)

    # Run inference with IFRNet
    with torch.no_grad():
        # Extract VGG features using the proper feature extractor
        vgg_features = vgg_feat_layer(image)
        
        # Get the last feature map from VGG (typically the one before the classifier)
        # The model expects a specific feature map, not the entire dictionary
        last_feature = vgg16(image)
        
        # Run inference with IFRNet
        output, _ = model(image, last_feature)

    # Denormalize the output
    output = output * 0.5 + 0.5  # Reverse normalization: (x * std) + mean

    # Save the output image
    save_image(output, output_image_path)

def main():
    parser = argparse.ArgumentParser(description="Remove Instagram filters from an image using IFRNet.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output', type=str, required=True, help="Path to save the unfiltered image")
    parser.add_argument('--model', type=str, default='weights/ifrnet.pth', help="Path to the pretrained model (default: weights/ifrnet.pth)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image {args.input} not found")

    remove_filter(args.input, args.output, args.model)

if __name__ == "__main__":
    main()