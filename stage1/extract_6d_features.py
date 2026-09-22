import pandas as pd
import torch
import torch.nn.functional as F
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

import config

def load_image(url):
    """Helper function to download an image from a URL on the fly."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"  [Warning] Could not load image {url}: {e}")
        return Image.new('RGB', (224, 224), color='white')

def main():
    print("1. Loading Data...")
    df = pd.read_csv(config.DATA_PATH, encoding="utf-8-sig")
    
    print("2. Loading Fine-Tuned DOST-RoBERTa...")
    text_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    text_model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_DIR)
    text_model.eval() 

    print("3. Loading M-CLIP Pipeline (Multilingual Text + Vision)...")
    # M-CLIP handles the Tagalog/Taglish text
    text_encoder = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
    # Standard CLIP handles the images
    image_encoder = SentenceTransformer('clip-ViT-B-32')

    print("4. Extracting 6D Features...")
    features_list = []

    for index, row in df.iterrows():
        text = row["review_text"]
        img_url = row["image_url"]
        star_rating = row["star_rating"]
        
        # --- A. Text Probabilities (Dims 1-4 from DOST-RoBERTa) ---
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH)
        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
            
        prob_authentic, prob_deceptive, prob_liv, prob_irrelevant = probs

        # --- B. Image-Text Similarity (Dim 5) ---
        image = load_image(img_url)
        
        # Image goes to standard CLIP, Text goes to M-CLIP
        img_emb = image_encoder.encode(image, convert_to_tensor=True)
        text_emb = text_encoder.encode(text, convert_to_tensor=True)
        
        # Calculate Cosine Similarity (How well the Tagalog text matches the picture)
        similarity_score = util.cos_sim(img_emb, text_emb).item()

        # --- C. Compile the 6D Vector ---
        features_list.append({
            "review_text": text,
            "label_stage1": row["label_stage1"], 
            "dim1_prob_auth": round(prob_authentic, 4),
            "dim2_prob_dec": round(prob_deceptive, 4),
            "dim3_prob_liv": round(prob_liv, 4),
            "dim4_prob_irr": round(prob_irrelevant, 4),
            "dim5_clip_sim": round(similarity_score, 4),
            "dim6_star_rating": star_rating
        })
        print(f"  Processed review {index + 1}/{len(df)}")

    # 5. Save the Features
    features_df = pd.DataFrame(features_list)
    output_path = "data/6d_features.csv"
    features_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSuccess! 6D features extracted and saved to {output_path}")

if __name__ == "__main__":
    main()