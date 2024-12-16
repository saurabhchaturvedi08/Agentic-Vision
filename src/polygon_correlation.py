from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def correlate_polygons(image1_path, image2_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Prepare images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Generate embeddings
    inputs1 = processor(text=["orange polygon"], images=image1, return_tensors="pt", padding=True)
    inputs2 = processor(text=["orange polygon"], images=image2, return_tensors="pt", padding=True)

    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

    # Calculate similarity
    similarity = outputs1.logits_per_image @ outputs2.logits_per_image.T
    return similarity.item()
