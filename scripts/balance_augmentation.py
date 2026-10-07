"""Fill quality-class shortfalls and sparse ABSA targets using training paraphrases.

Offline, reproducible rule-based augmentation, not an LLM or human annotation.
Original CSV and split manifest are never rewritten. Outputs remain pending.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from model_contract import ASPECTS, CLASS_NAMES, DEFAULT_DATA_PATH, DEFAULT_SPLITS_PATH, POLARITIES, map_polarity
from model_data import load_experiment, fingerprint
from scripts.prepare_augmentation import balancing_report
from training_augmentation import AUGMENTATION_COLUMNS, append_augmentations, image_scopes

# Alternatives preserve the original claim. They add no new product experience.
# Replacement is simultaneous, so an inserted alternative is never rewritten.
ALTERNATIVES = {
    "absolutely thrilled": ("absolutely thrilled", "extremely happy", "very pleased", "really delighted"),
    "exceeded my expectations": ("exceeded my expectations", "went beyond my expectations", "surpassed my expectations"),
    "in every way": ("in every way", "in all respects"),
    "sleek design": ("sleek design", "sleek appearance", "stylish design"),
    "impressive performance": ("impressive performance", "excellent performance", "great performance"),
    "game-changer": ("game-changer", "game changer", "major improvement"),
    "top-notch": ("top-notch", "excellent", "high quality", "first-rate"),
    "attention to detail": ("attention to detail", "careful attention to details"),
    "setup was a breeze": ("setup was a breeze", "setup was easy", "setting it up was easy", "setup was straightforward"),
    "user-friendly interface": ("user-friendly interface", "easy-to-use interface", "simple-to-use interface"),
    "makes operation a delight": ("makes operation a delight", "makes it enjoyable to operate", "makes operation enjoyable"),
    "significant improvement": ("significant improvement", "noticeable improvement", "considerable improvement"),
    "overall experience": ("overall experience", "experience overall"),
    "helpful assistance": ("helpful assistance", "helpful support", "useful assistance"),
    "whenever needed": ("whenever needed", "whenever I needed it"),
    "daily life": ("daily life", "everyday life"),
    "highly recommend": ("highly recommend", "strongly recommend", "definitely recommend"),
    "wrong item delivered": ("wrong item delivered", "the wrong item was delivered", "received the wrong item"),
    "leaves skin feeling fresh": ("leaves skin feeling fresh", "leaves the skin feeling fresh", "makes skin feel fresh"),
    "improves skin elasticity": ("improves skin elasticity", "improves the skin's elasticity", "helps skin elasticity improve"),
    "smoother skin texture": ("smoother skin texture", "smoother texture of the skin", "skin with a smoother texture"),
    "perfect for daily use": ("perfect for daily use", "perfect for everyday use", "perfect to use daily"),
    "evens out skin": ("evens out skin", "makes skin more even", "helps even out skin"),
    "refreshing lemon scent": ("refreshing lemon scent", "refreshing lemon fragrance", "fresh lemon scent", "refreshing scent of lemon"),
    "my recent purchase": ("my recent purchase", "my latest purchase", "the purchase I recently made"),
    "customer service": ("customer service", "customer support"),
    "a product must buy": ("a product must buy", "a must-buy product", "a product I strongly recommend buying", "a product worth buying", "a product you should buy"),
    "I'm giving it a 1 star": ("I'm giving it a 1 star", "I'm rating it one star", "I'm giving it a one-star rating"),
    "hours before I received this": ("hours before I received this", "a few hours before this arrived", "hours before this was delivered to me"),
    "my mom died": ("my mom died", "my mother passed away", "my mom passed away"),
    "the products quality is 11/10": ("the products quality is 11/10", "the product quality is 11 out of 10", "I'd give the product quality 11/10"),
    "there were no damages": ("there were no damages", "there was no damage", "nothing was damaged"),
    "this is my 3rd time": ("this is my 3rd time", "this is my third time"),
    "the item was packaged well": ("the item was packaged well", "the item was well packaged", "the product was packed well"),
    "I love the design": ("I love the design", "I really like the design", "I like the design a lot"),
    "I highly recommend it": ("I highly recommend it", "I strongly recommend it", "I definitely recommend it"),
    "maganda nman": ("maganda nman", "maganda naman", "nice naman"),
    "mura pa": ("mura pa", "affordable pa", "mura rin"),
    "di din nagbubuhol": ("di din nagbubuhol", "hindi rin nagbubuhol", "hindi din nagbubuhol"),
    "makapal ang wire": ("makapal ang wire", "thick ang wire", "makapal yung wire"),
    "nagustohan ng anak ko": ("nagustohan ng anak ko", "nagustuhan ng anak ko", "nagustuhan naman ng anak ko"),
    "super effective": ("super effective", "sobrang effective", "very effective", "highly effective"),
    "bili na kayo": ("bili na kayo", "bumili na kayo", "buy na kayo", "bilhin na ninyo"),
    "napamura ako sa tuwa": ("napamura ako sa tuwa", "napamura ako sa sobrang tuwa", "sa tuwa ko ay napamura ako"),
    "best feature": ("best feature", "best feature of this", "its best feature"),
    "value for money": ("value for money", "worth the price", "value at this price"),
    "well packed": ("well packed", "well packaged", "securely packed"),
    "not worth it": ("not worth it", "not worth the price", "hindi worth it"),
    "thank you": ("thank you", "thanks", "maraming salamat"),
    "thankyou": ("thankyou", "thank you", "thanks"),
    "thanks": ("thanks", "thank you", "salamat"),
    "salamat": ("salamat", "thank you", "thanks"),
    "delivery man": ("delivery man", "delivery rider", "delivery person"),
    "delivery guy": ("delivery guy", "delivery rider", "delivery person"),
    "delivery time": ("delivery time", "time taken for delivery"),
    "shipping": ("shipping", "shipment"),
    "parcel": ("parcel", "package"),
    "I received": ("I received", "I got"),
    "delivered": ("delivered", "brought"),
    "I purchased": ("I purchased", "I bought"),
    "item": ("item", "product"),
    "product": ("product", "item"),
    "seller": ("seller", "shop"),
    "shop": ("shop", "seller"),
    "nice": ("nice", "good", "maganda"),
    "excellent": ("excellent", "great", "very good"),
    "great": ("great", "very good", "excellent"),
    "good": ("good", "nice", "maganda"),
    "maganda": ("maganda", "nice", "good"),
    "ganda": ("ganda", "maganda", "nice"),
    "pangit": ("pangit", "ugly"),
    "poor": ("poor", "bad"),
    "bad": ("bad", "poor"),
    "sulit": ("sulit", "worth it"),
    "mabilis": ("mabilis", "fast", "quick"),
    "fast": ("fast", "quick", "mabilis"),
    "quick": ("quick", "fast"),
    "fragrance": ("fragrance", "scent"),
    "smell": ("smell", "scent"),
    "comfortable": ("comfortable", "comfy"),
    "comfy": ("comfy", "comfortable"),
    "price": ("price", "cost", "presyo"),
    "expensive": ("expensive", "pricey"),
    "affordable": ("affordable", "budget-friendly"),
    "cheap": ("cheap",),
    "perfect": ("perfect", "excellent"),
    "happy": ("happy", "pleased"),
    "magaan": ("magaan", "lightweight"),
    "lightweight": ("lightweight", "magaan"),
    "hindi": ("hindi", "di"),
    "hnd": ("hnd", "hindi"),
    "naman": ("naman", "nmn"),
    "nmn": ("nmn", "naman"),
    "yung": ("yung", "iyong"),
    "ung": ("ung", "yung"),
    "yun": ("yun", "iyon"),
    "siya": ("siya", "sya"),
    "sya": ("sya", "siya"),
    "talaga": ("talaga", "tlga"),
    "tlga": ("tlga", "talaga"),
    "sobrang": ("sobrang", "super"),
    "super": ("super", "sobrang"),
}
ALTERNATIVES = {key.casefold(): values for key, values in ALTERNATIVES.items()}
ALTERNATIVES.update({
    "get a chance to win": ("get a chance to win", "have a chance to win", "join for a chance to win", "enter for a chance to win"),
    "all-expense paid trip": ("all-expense paid trip", "all-expenses-paid trip", "trip with all expenses paid"),
    "just buy": ("just buy", "purchase", "simply buy"),
    "pay using gcash": ("pay using GCash", "pay through GCash", "use GCash to pay", "make payment with GCash"),
    "join now": ("join now", "enter now", "take part now"),
    "pick your prize": ("pick your prize", "choose your prize", "select your prize"),
    "promo page": ("promo page", "promotion page", "page for the promo"),
    "for free": ("for free", "at no charge", "without a charge"),
    "no advisories": ("no advisories", "don't want advisories", "don't want the notices"),
    "brand new car": ("brand new car", "new car", "brand-new car"),
    "ongoing investigation": ("ongoing investigation", "investigation currently underway", "current investigation"),
    "trust fund remains untouched": ("trust fund remains untouched", "trust fund has not been touched", "trust fund is still untouched"),
    "concerns surrounding": ("concerns surrounding", "concerns about", "concerns regarding"),
    "my neighbor": ("my neighbor", "the person next door", "my next-door neighbor"),
    "txt or call me": ("txt or call me", "text or call me", "send a message or call me"),
    "I'm working at home": ("I'm working at home", "I work from home", "I'm working from home"),
    "I deserve to be notified": ("I deserve to be notified", "I should be notified", "I deserve a notification"),
    "came from overseas": ("came from overseas", "was sent from overseas", "came from abroad"),
    "looks cheap": ("looks cheap", "looks low-quality", "looks cheaply made"),
    "two times": ("two times", "twice"),
    "sa bahay ng kapatid ko": ("sa bahay ng kapatid ko", "sa bahay ng aking kapatid", "sa tahanan ng kapatid ko"),
    "nakalagay na address": ("nakalagay na address", "address na nilagay", "address na ibinigay"),
    "hindi binalik": ("hindi binalik", "di ibinalik", "hindi ibinalik"),
    "hindi naman": ("hindi naman", "di naman"),
    "wala nmang": ("wala nmang", "wala namang"),
    "itinapon": ("itinapon", "inihagis"),
    "tumawag": ("tumawag", "nag-call"),
    "tawag": ("tawag", "call"),
    "text": ("text", "message"),
    "natanggap": ("natanggap", "nareceive"),
    "narecieved": ("narecieved", "natanggap", "nareceive"),
    "naconfuse": ("naconfuse", "nalito"),
    "diko sure": ("diko sure", "hindi ako sure", "di ko sigurado"),
    "nag deliver": ("nag deliver", "nag-deliver", "naghatid"),
    "magdeliver": ("magdeliver", "mag-deliver"),
    "nag hintay": ("nag hintay", "naghintay", "nag-antay"),
    "mag hapon": ("mag hapon", "maghapon", "buong araw"),
    "gabi na": ("gabi na", "nighttime na"),
    "couries": ("couries", "courier"),
    "palagi": ("palagi", "lagi"),
    "lagi": ("lagi", "palagi"),
    "paggagamitan": ("paggagamitan", "gagamitin"),
    "pagkadating": ("pagkadating", "pagdating"),
    "nakaka dismaya": ("nakaka dismaya", "nakakadismaya", "nakakadisappoint"),
    "nakaka disapoint": ("nakaka disapoint", "nakakadismaya", "nakakadisappoint"),
})
ALTERNATIVES = {key.casefold(): values for key, values in ALTERNATIVES.items()}
TOKEN_RE = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(key) for key in sorted(ALTERNATIVES, key=len, reverse=True)) + r")(?!\w)", re.I)


def lexical_key(text):
    """Punctuation/case-only changes are not new paraphrases."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def source_annotation_issue(source):
    """Avoid multiplying known conflicts with the agreed annotation definitions."""
    if int(source["label"]) != 0:
        return None
    title = source.get("product_title", "").casefold()
    clearly_non_beauty = re.search(r"mouse|earphone|earbud|headset|speaker|monitor|webcam|shoes|pants", title)
    for annotation in json.loads(source["aspect_annotations"]):
        category, evidence = annotation["category"], annotation["text"]
        if category == "sensory_experience" and clearly_non_beauty:
            return "sensory_experience is restricted to beauty/personal-care products"
        if category == "accuracy_of_description" and map_polarity(annotation["sentiment"]) == 1:
            if not re.search(r"description|advertis|picture|listed|as shown|specification|size chart|nakalagay", evidence, re.I):
                return "neutral accuracy evidence lacks a reviewer-stated advertised/received comparison"
        if category == "seller_service" and evidence.rstrip().endswith("?"):
            return "a request/question alone does not establish concrete seller behavior"
        if evidence not in source["review_text"]:
            return "source evidence is not an exact substring; reannotation needed before automatic paraphrasing"
    return None


def variant(source, rng):
    raw = source["review_text"]
    # Every occurrence of the same source phrase gets the same replacement,
    # so a transformed annotation stays identical to its transformed evidence.
    choices = {}
    for match in TOKEN_RE.finditer(raw):
        key = match.group().lower()
        if key not in choices:
            options = ALTERNATIVES[key]
            choices[key] = rng.choice(options)

    def transform(text):
        return TOKEN_RE.sub(lambda match: choices.get(match.group().lower(), match.group()), text)

    text = transform(raw)
    annotations = []
    if int(source["label"]) == 0:
        grouped = {}
        for annotation in json.loads(source["aspect_annotations"]):
            evidence = transform(annotation["text"])
            if evidence not in text:
                return None
            category, polarity = annotation["category"], map_polarity(annotation["sentiment"])
            candidate = {"category": category, "text": evidence, "sentiment": POLARITIES[polarity].lower()}
            # One representative exact span per category/polarity.
            if (category, polarity) not in grouped or len(evidence) > len(grouped[(category, polarity)]["text"]):
                grouped[(category, polarity)] = candidate
        annotations = list(grouped.values())
    # Move independently tagged fields; keep narrative/review body at the end.
    blocks = re.split(r"\s*\|\s*", text)
    if len(blocks) > 2 and all(any(annotation["text"] in block for block in blocks) for annotation in annotations):
        fields, narrative = blocks[:-1], blocks[-1]
        rng.shuffle(fields)
        text = " | ".join([*fields, narrative])
    if any(annotation["text"] not in text for annotation in annotations):
        return None
    return text, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--augmentations", default="data/augmented_reviews.csv")
    parser.add_argument("--base-augmentations", help="Optional earlier draft CSV to expand; output still goes to --augmentations")
    parser.add_argument("--aspect-minimum", type=int, default=30)
    parser.add_argument("--quality-target", type=int, help="Minimum per quality class; final target also accommodates added Authentic ABSA rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="reports/generated/augmentation_full_balance.json")
    args = parser.parse_args()
    if args.aspect_minimum < 0 or (args.quality_target is not None and args.quality_target < 1):
        raise ValueError("Targets must be nonnegative (quality target positive)")
    original = load_experiment(args.data, args.splits, require_annotations=True)
    path = Path(args.augmentations)
    base_path = Path(args.base_augmentations) if args.base_augmentations else path
    draft = pd.read_csv(base_path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if base_path.exists() else pd.DataFrame(columns=AUGMENTATION_COLUMNS)
    # Existing drafts are preserved. Their acceptance is projected, not persisted.
    projection = draft.copy()
    pending = projection.review_status.eq("pending")
    projection.loc[pending, "review_status"] = "approved"
    projection.loc[pending, "reviewed_by"] = "structural-projection-only"
    projection.loc[pending & projection.review_notes.eq(""), "review_notes"] = "Projection only"
    projected = append_augmentations(original, projection)
    train = original[original.partition.eq("train")]
    all_sources = train.to_dict("records")
    excluded_sources = [{"source_review_id": row["review_id"], "reason": issue}
                        for row in all_sources if (issue := source_annotation_issue(row))]
    excluded_ids = {row["source_review_id"] for row in excluded_sources}
    sources = [row for row in all_sources if row["review_id"] not in excluded_ids]
    scopes = image_scopes(original)
    seen = {lexical_key(text) for text in itertools.chain(original.review_text, draft.review_text)}
    ids = set(original.review_id) | set(draft.review_id)
    source_use = Counter(draft.loc[draft.review_status.ne("rejected"), "source_review_id"])
    qualities = Counter(projected.loc[projected.partition.eq("train"), "label"])
    aspects = Counter()
    for annotations in projected.loc[projected.partition.eq("train") & projected.label.eq(0), "annotations"]:
        aspects.update({(item["category"], item["sentiment"]) for item in annotations})
    rng = random.Random(args.seed)
    rows, exhausted = [], set()
    source_pairs = {row["review_id"]: {(item["category"], item["sentiment"]) for item in row["annotations"]}
                    for row in sources if row["label"] == 0}

    def add_from(candidates):
        # Prefer the least reused real source before trying richer alternatives.
        candidates = sorted(candidates, key=lambda row: (source_use[row["review_id"]], row["fold"], row["product_id"], row["review_id"]))
        for source in candidates:
            source_id = source["review_id"]
            if source_id in exhausted:
                continue
            for _ in range(160):
                result = variant(source, rng)
                if result is None:
                    continue
                text, annotations = result
                key = lexical_key(text)
                if key in seen:
                    continue
                row_id = "synthetic_full_" + hashlib.sha256((source_id + "\n" + text).encode()).hexdigest()[:16]
                if row_id in ids:
                    continue
                safe = bool(source["image_urls"]) and all(scopes[url] == {("train", int(source["fold"]))} for url in source["image_urls"])
                label = int(source["label"])
                note = "Rule-based paraphrase (seed=%s), pending human review; inherited source quality label, product context, rating and OOF fold. " % args.seed
                if label == 1:
                    note += "Deceptive source label requires corroboration; rating/image mismatch or generic praise alone is insufficient. "
                if label == 0:
                    note += "All source category/polarity pairs preserved; exact transformed evidence checked. Verify the source taxonomy and polarity labels. "
                note += "Inspect reused source photos." if safe else "No photos selected because URLs are absent or shared across scopes; confirm label without visual support."
                rows.append(dict(review_id=row_id, source_review_id=source_id, review_text=text,
                    ground_truth=CLASS_NAMES[label], aspect_annotations=json.dumps(annotations, ensure_ascii=False),
                    image_mode="source" if safe else "none", image_source_review_id="",
                    augmentation_method="paraphrase", review_status="pending", reviewed_by="", review_notes=note))
                seen.add(key); ids.add(row_id); source_use[source_id] += 1; qualities[label] += 1
                if label == 0:
                    aspects.update(source_pairs[source_id])
                return True
            exhausted.add(source_id)
        return False

    unfillable = []
    for pair in sorted([(category, index) for category in ASPECTS for index in range(3)], key=lambda pair: aspects[pair]):
        candidates = [row for row in sources if row["label"] == 0 and pair in source_pairs[row["review_id"]]]
        while aspects[pair] < args.aspect_minimum:
            if not add_from(candidates):
                unfillable.append({"category": pair[0], "sentiment": POLARITIES[pair[1]].lower(), "count": aspects[pair], "target": args.aspect_minimum})
                break
    target = max([args.quality_target or 0, *qualities.values()])
    for label in (1, 3, 2, 0):
        candidates = [row for row in sources if row["label"] == label]
        while qualities[label] < target:
            if not add_from(candidates):
                raise ValueError(f"Cannot generate enough distinct meaning-preserving {CLASS_NAMES[label]} variants: {qualities[label]}/{target}. No output overwritten.")
        print(f"{CLASS_NAMES[label]} projected training count: {qualities[label]}", flush=True)
    next_draft = pd.concat([draft, pd.DataFrame(rows, columns=AUGMENTATION_COLUMNS)], ignore_index=True)
    validation = next_draft.copy()
    pending = validation.review_status.eq("pending")
    validation.loc[pending, "review_status"] = "approved"
    validation.loc[pending, "reviewed_by"] = "structural-check-only"
    verified = append_augmentations(original, validation)
    assert Counter(verified.loc[verified.partition.eq("train"), "label"]) == Counter({label: target for label in range(4)})
    assert set(verified.loc[verified.partition.ne("train"), "review_id"]) == set(original.loc[original.partition.ne("train"), "review_id"])
    active = append_augmentations(original, next_draft)
    # Keep an exact local backup before replacing the augmentation file.
    backup = Path(args.report).with_suffix(".previous.csv")
    if rows and path.exists():
        if backup.exists():
            raise FileExistsError(f"Backup exists: {backup}; choose a new --report name")
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(path.read_bytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    next_draft.to_csv(path, index=False, encoding="utf-8-sig")
    report = balancing_report(original, active, next_draft)
    report["batch_policy"] = {
        "mode": "full training-set balance; supersedes the small-batch pilot limits",
        "quality_target_per_class": target,
        "aspect_minimum": args.aspect_minimum,
        "source_variants": "Not capped; reported explicitly and remain pending human review",
        "held_out_data": "Original validation and test rows only",
    }
    report["full_balance_generation"] = {
        "method": "offline rule-based lexical paraphrases and independent tagged-field ordering; not an LLM",
        "seed": args.seed, "new_rows": len(rows), "quality_target": target,
        "aspect_minimum": args.aspect_minimum, "unfillable_aspect_targets": unfillable,
        "source_annotation_exclusions": excluded_sources,
        "max_variants_per_source": max(source_use.values(), default=0),
        "source_coverage_by_class": {CLASS_NAMES[label]: {"original_reviews": int(train.label.eq(label).sum()),
            "original_products": int(train.loc[train.label.eq(label), "product_id"].nunique())} for label in range(4)},
        "approval": "New rows are pending, with no human reviewer recorded; projected balance is not active training balance",
        "original_dataset_sha256": fingerprint(original),
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(rows)} new drafts; {len(next_draft)} total. Projected train: {target} per class. Report: {output}")
    if unfillable:
        print("Some ABSA targets lack distinct eligible paraphrases:", json.dumps(unfillable))


if __name__ == "__main__":
    main()
