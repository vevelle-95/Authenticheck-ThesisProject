"""Training-only augmentation rules shared by AuthentiCheck and its baseline.

Validate the original CSV against its split manifest BEFORE calling this module.
It uses no model libraries and never changes the real split manifest.
"""

import json
from collections import defaultdict

import pandas as pd

from model_contract import ASPECTS, map_polarity, map_quality, parse_image_urls

AUGMENTATION_COLUMNS = (
    "review_id", "source_review_id", "review_text", "ground_truth",
    "aspect_annotations", "image_mode", "image_source_review_id",
    "augmentation_method", "review_status", "reviewed_by", "review_notes",
)
PROVENANCE_COLUMNS = (
    "source_review_id", "image_source_review_id", "image_mode",
    "augmentation_method", "reviewed_by", "review_notes", "review_status",
)


def text_key(text):
    return " ".join(text.split()).casefold()


def image_scopes(assigned):
    scopes = defaultdict(set)
    for row in assigned.itertuples():
        for url in parse_image_urls(row.review_image_urls):
            scopes[url].add((row.partition, int(row.fold)))
    return scopes


def annotation_pairs(raw, text, *, exact=False):
    annotations = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(annotations, list):
        raise ValueError("aspect_annotations must be a JSON list, including [] for none")
    pairs = set()
    for item in annotations:
        if not isinstance(item, dict) or item.get("category") not in ASPECTS:
            raise ValueError("Unknown aspect category")
        evidence = item.get("text")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Aspect evidence must be nonempty")
        if exact and (set(item) != {"category", "text", "sentiment"} or evidence not in text):
            raise ValueError("New aspect evidence must be an exact review substring with category/text/sentiment fields")
        if exact and item.get("sentiment") not in ("negative", "neutral", "positive"):
            raise ValueError("New aspect sentiments must be lowercase negative, neutral, or positive")
        pairs.add((item["category"], map_polarity(item.get("sentiment"))))
    return pairs


def append_augmentations(assigned, path=None, *, allow_unreviewed=False):
    """Append approved rows, or explicitly opt into pending experimental rows.

    Images are resolved from original rows, never from LLM-created URL strings.
    Pending rows keep their unreviewed status when opted in. Rejected rows are
    always excluded. All source, fold, image and annotation checks still apply.
    """
    if path is None:
        if allow_unreviewed:
            raise ValueError("Unreviewed augmentation mode requires an augmentation CSV")
        return assigned
    draft = (path.copy() if isinstance(path, pd.DataFrame) else
             pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype=str))
    missing = set(AUGMENTATION_COLUMNS) - set(draft)
    extra = set(draft) - set(AUGMENTATION_COLUMNS)
    if missing or extra:
        raise ValueError(f"Augmentation CSV columns: missing={sorted(missing)}, unexpected={sorted(extra)}")
    draft = draft.apply(lambda column: column.str.strip())
    if not draft.review_status.isin(("pending", "approved", "rejected")).all():
        raise ValueError("review_status must be pending, approved, or rejected")
    if draft.review_id.eq("").any() or draft.review_id.duplicated().any():
        raise ValueError("Every augmentation draft needs a unique nonempty review_id")
    if set(draft.review_id) & set(assigned.review_id):
        raise ValueError("Augmentation IDs must not overlap original review IDs")
    if "source_review_id" in assigned:
        raise ValueError("Apply augmentation once, to the original assigned reviews")
    selected = draft[draft.review_status.isin(("approved", "pending") if allow_unreviewed else ("approved",))]
    if selected.empty:
        return assigned
    sources = assigned.set_index("review_id", drop=False)
    scopes = image_scopes(assigned)
    seen_text = set(assigned.review_text.map(text_key))
    rows = []
    for entry in selected.to_dict("records"):
        try:
            if entry["source_review_id"] not in sources.index:
                raise ValueError("source_review_id must identify an original review")
            source = sources.loc[entry["source_review_id"]]
            if source.partition != "train":
                raise ValueError("Augmentation sources must belong to training, never validation/test")
            if entry["review_status"] == "approved" and (not entry["reviewed_by"] or not entry["review_notes"]):
                raise ValueError("Approved rows need reviewed_by and review_notes confirming the label and image/evidence checks")
            if not entry["review_text"] or text_key(entry["review_text"]) in seen_text:
                raise ValueError("Generated text must be nonempty and distinct from all original/approved reviews")
            label = map_quality(source.ground_truth)
            if map_quality(entry["ground_truth"]) != label:
                raise ValueError("Preserve the source quality label; review source-label disputes separately")
            method = entry["augmentation_method"]
            if method not in ("paraphrase", "controlled_image_mismatch"):
                raise ValueError("augmentation_method must be paraphrase or controlled_image_mismatch")
            if method == "controlled_image_mismatch" and (label not in (1, 3) or entry["image_mode"] != "donor"):
                raise ValueError("Controlled image mismatches require a deceptive/irrelevant source and donor image mode")
            pairs = annotation_pairs(entry["aspect_annotations"], entry["review_text"], exact=True)
            if label == 0 and pairs != annotation_pairs(source.aspect_annotations, source.review_text):
                raise ValueError("Authentic paraphrases must preserve all source aspect/sentiment pairs, including mixed sentiments")
            mode = entry["image_mode"]
            donor_id = entry["image_source_review_id"]
            if mode == "none":
                if donor_id:
                    raise ValueError("image_source_review_id must be blank for image_mode=none")
                urls = []
            elif mode in ("source", "donor"):
                if mode == "source":
                    if donor_id and donor_id != source.review_id:
                        raise ValueError("Source image mode must use the text source's images")
                    donor_id = source.review_id
                elif method != "controlled_image_mismatch":
                    raise ValueError("Donor mode requires controlled_image_mismatch and human review")
                if donor_id not in sources.index:
                    raise ValueError("Image donor must identify an original review")
                donor = sources.loc[donor_id]
                if donor.partition != "train" or int(donor.fold) != int(source.fold):
                    raise ValueError("Image donor must be training-only and in the source OOF fold")
                if mode == "donor" and donor_id == source.review_id:
                    raise ValueError("Donor mode requires a different original review")
                urls = parse_image_urls(donor.review_image_urls)
                if not urls:
                    raise ValueError("Image source has no URLs; use image_mode=none")
                if any(scopes[url] != {("train", int(source.fold))} for url in urls):
                    raise ValueError("A reused image URL also occurs in another fold or validation/test; select another source or image_mode=none")
            else:
                raise ValueError("image_mode must be source, donor, or none")
            row = source.to_dict()
            row.update({key: entry[key] for key in PROVENANCE_COLUMNS})
            row.update(
                review_id=entry["review_id"], review_text=entry["review_text"],
                aspect_annotations=entry["aspect_annotations"],
                review_image_urls=json.dumps(urls, ensure_ascii=False), image_urls=urls,
                image_source_review_id=donor_id,
            )
            rows.append(row)
            seen_text.add(text_key(entry["review_text"]))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Augmentation {entry['review_id']}: {error}") from error
    originals = assigned.copy()
    for column in PROVENANCE_COLUMNS:
        originals[column] = ""
    return pd.concat([originals, pd.DataFrame(rows)], ignore_index=True)


def augmentation_summary(frame):
    """Record what was actually used, without claiming human review of drafts."""
    if "source_review_id" not in frame:
        return {"approved_rows": 0, "unreviewed_rows": 0}
    generated = frame[frame.source_review_id.ne("")]
    return {
        "approved_rows": int(generated.review_status.eq("approved").sum()),
        "unreviewed_rows": int(generated.review_status.eq("pending").sum()),
    }
