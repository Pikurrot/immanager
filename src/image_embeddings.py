import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_images_from_folder(folder_path):
    images = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            images[filename] = Image.open(path).convert("RGB")
    return images

def get_image_embeddings(images_dict):
    embeddings = {}
    for filename, img in images_dict.items():
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
        embeddings[filename] = embedding.cpu().numpy()
    return embeddings

def get_text_embedding(text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_input)
    return text_embedding.cpu().numpy()
