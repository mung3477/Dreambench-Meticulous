import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def evaluate_clip_similarity(image_path1: str, image_path2: str) -> float:
    """
    Evaluates the cosine similarity between two images using the CLIP model.
    
    Args:
        image_path1 (str): Filepath to the first image.
        image_path2 (str): Filepath to the second image.
        
    Returns:
        float: Cosine similarity score between the image embeddings.
    """
    # Detect the best available hardware device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pre-trained CLIP model and processor from Hugging Face
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Load images and ensure they are in RGB format
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    # Preprocess images (resizing, normalization, etc.) and move to the device
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    inputs2 = processor(images=image2, return_tensors="pt").to(device)

    # Extract image features without computing gradients
    with torch.no_grad():
        features1 = model.get_image_features(**inputs1)
        features2 = model.get_image_features(**inputs2)

    # L2 normalize the feature vectors before computing cosine similarity
    features1 = F.normalize(features1, p=2, dim=-1)
    features2 = F.normalize(features2, p=2, dim=-1)

    # Compute cosine similarity
    similarity = F.cosine_similarity(features1, features2)
    
    return similarity.item()

if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Evaluate image similarity using CLIP")
    parser.add_argument("image1", type=str, help="Path to the first image")
    parser.add_argument("image2", type=str, help="Path to the second image")
    
    args = parser.parse_args()
    
    # Calculate and print the similarity score
    score = evaluate_clip_similarity(args.image1, args.image2)
    print(f"CLIP Similarity Score: {score:.4f}")
