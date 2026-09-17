import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

def evaluate_dino_similarity(image_path1: str, image_path2: str) -> float:
    """
    Evaluates the cosine similarity between two images using the DINOv2 model.
    
    Args:
        image_path1 (str): Filepath to the first image.
        image_path2 (str): Filepath to the second image.
        
    Returns:
        float: Cosine similarity score between the image embeddings.
    """
    # Detect the best available hardware device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pre-trained DINOv2 model and processor from Hugging Face
    model_id = "facebook/dinov2-base"
    model = AutoModel.from_pretrained(model_id).to(device)
    processor = AutoImageProcessor.from_pretrained(model_id)

    # Load images and ensure they are in RGB format
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    # Preprocess images (resizing, normalization, etc.) and move to the device
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    inputs2 = processor(images=image2, return_tensors="pt").to(device)

    # Extract image features without computing gradients
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # We use the CLS token representation (first token in the sequence) as the global image embedding
    features1 = outputs1.last_hidden_state[:, 0, :]
    features2 = outputs2.last_hidden_state[:, 0, :]

    # L2 normalize the feature vectors before computing cosine similarity
    features1 = F.normalize(features1, p=2, dim=-1)
    features2 = F.normalize(features2, p=2, dim=-1)

    # Compute cosine similarity
    similarity = F.cosine_similarity(features1, features2)
    
    return similarity.item()

if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Evaluate image similarity using DINOv2")
    parser.add_argument("image1", type=str, help="Path to the first image")
    parser.add_argument("image2", type=str, help="Path to the second image")
    
    args = parser.parse_args()
    
    # Calculate and print the similarity score
    score = evaluate_dino_similarity(args.image1, args.image2)
    print(f"DINOv2 Similarity Score: {score:.4f}")
