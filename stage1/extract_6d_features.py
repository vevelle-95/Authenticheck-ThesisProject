import pandas as pd
import requests
import torch
import torch.nn.functional as F
from io import BytesIO
from pathlib import Path

from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


LABEL_MAP = {
    "authentic": 0,
    "deceptive": 1,
    "vague": 2,
    "low informational value": 2,
    "low_informational_value": 2,
    "low-value": 2,
    "low_value": 2,
    "irrelevant": 3,
}


def map_ground_truth(value):
    if pd.isna(value):
        raise ValueError("ground_truth cannot be empty.")

    normalized = str(value).strip().lower()

    if normalized in LABEL_MAP:
        return LABEL_MAP[normalized]

    try:
        numeric = int(float(normalized))
    except ValueError as error:
        raise ValueError(
            f"Unknown ground_truth label: {value!r}"
        ) from error

    if numeric not in range(4):
        raise ValueError(
            f"ground_truth must map to 0, 1, 2, or 3: {value!r}"
        )

    return numeric


def normalize_rating(rating):
    if rating is None or pd.isna(rating):
        rating = 3.0

    return max(
        0.0,
        min(1.0, (float(rating) - 1.0) / 4.0),
    )


def parse_image_urls(value):
    if value is None or pd.isna(value):
        return []

    return [
        url.strip()
        for url in str(value).split("|")
        if url.strip()
    ]


def load_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(
            BytesIO(response.content)
        ).convert("RGB")
    except Exception as error:
        print(f"  [Warning] Could not load image {url}: {error}")
        return None


def build_context(product_description, review_text):
    product_description = str(product_description or "").strip()
    review_text = str(review_text or "").strip()

    return product_description, review_text


def extract_image_scores(
    image_urls,
    image_encoder,
    text_embedding,
):
    scores = []
    valid_urls = []

    for image_url in image_urls:
        image = load_image(image_url)

        if image is None:
            continue

        image_embedding = image_encoder.encode(
            image,
            convert_to_tensor=True,
        )

        score = float(
            util.cos_sim(
                image_embedding,
                text_embedding,
            ).item()
        )

        scores.append(score)
        valid_urls.append(image_url)

    return valid_urls, scores


def main():
    if not config.DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {config.DATA_PATH}.")
        return

    print("1. Loading root dataset...")
    df = pd.read_csv(
        config.DATA_PATH,
        encoding="utf-8-sig",
        skipinitialspace=True,
    )

    required_columns = {
        "review_text",
        "product_description",
        "review_image_urls",
        "star_rating",
        "ground_truth",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(f"   Loaded {len(df)} reviews.")

    print("2. Loading fine-tuned DOST-RoBERTa...")
    text_tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_DIR,
        use_fast=True,
    )
    text_model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_DIR
    )
    text_model.eval()

    print("3. Loading CLIP models...")
    text_encoder = SentenceTransformer(
        "clip-ViT-B-32-multilingual-v1"
    )
    image_encoder = SentenceTransformer(
        "clip-ViT-B-32"
    )

    print("4. Extracting 6D features...")
    features_list = []

    for index, row in df.iterrows():
        product_description, review_text = build_context(
            row["product_description"],
            row["review_text"],
        )

        if not review_text:
            raise ValueError(
                f"Review at row {index} is empty."
            )

        ground_truth = map_ground_truth(
            row["ground_truth"]
        )
        normalized_rating = normalize_rating(
            row["star_rating"]
        )

        # Use the same product-description/review pair
        # used during Stage 1 training.
        inputs = text_tokenizer(
            product_description,
            review_text,
            return_tensors="pt",
            padding="max_length",
            truncation="only_first",
            max_length=config.MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = text_model(**inputs)
            probabilities = F.softmax(
                outputs.logits,
                dim=-1,
            ).squeeze(0).tolist()

        if len(probabilities) != 4:
            raise ValueError(
                "Stage 1 model must output exactly four probabilities."
            )

        (
            prob_authentic,
            prob_deceptive,
            prob_liv,
            prob_irrelevant,
        ) = probabilities

        # Use the contextual text for the image-text comparison.
        combined_text = (
            f"{product_description}\n{review_text}"
            if product_description
            else review_text
        )

        text_embedding = text_encoder.encode(
            combined_text,
            convert_to_tensor=True,
        )

        image_urls = parse_image_urls(
            row["review_image_urls"]
        )

        valid_urls, image_scores = extract_image_scores(
            image_urls,
            image_encoder,
            text_embedding,
        )

        # Select the most similar useful image.
        if image_scores:
            best_image_index = int(
                max(
                    range(len(image_scores)),
                    key=lambda item: image_scores[item],
                )
            )
            similarity_score = image_scores[best_image_index]
            best_image_url = valid_urls[best_image_index]
        else:
            similarity_score = 0.0
            best_image_url = ""

        features_list.append(
            {
                "review_text": review_text,
                "product_description": product_description,
                "ground_truth": ground_truth,
                "dim1_prob_auth": round(
                    prob_authentic,
                    4,
                ),
                "dim2_prob_dec": round(
                    prob_deceptive,
                    4,
                ),
                "dim3_prob_liv": round(
                    prob_liv,
                    4,
                ),
                "dim4_prob_irr": round(
                    prob_irrelevant,
                    4,
                ),
                "dim5_clip_sim": round(
                    similarity_score,
                    4,
                ),
                "dim6_star_rating": round(
                    normalized_rating,
                    4,
                ),
                "image_scores": [
                    round(score, 4)
                    for score in image_scores
                ],
                "best_image_url": best_image_url,
            }
        )

        print(
            f"  Processed review "
            f"{index + 1}/{len(df)} "
            f"({len(image_scores)} valid images)"
        )

    print("5. Saving six-dimensional features...")
    features_df = pd.DataFrame(features_list)

    output_path = Path(config.FEATURES_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Success! Features saved to {output_path}")


if __name__ == "__main__":
    main()