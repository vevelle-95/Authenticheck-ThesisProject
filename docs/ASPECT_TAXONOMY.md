# Fixed aspect taxonomy

The following ten categories are the agreed AuthentiCheck taxonomy, as specified by the research team. This supersedes the earlier five-category taxonomy in the supplied proposal for future implementation and annotation. The thesis paper must be updated to describe this same taxonomy.

Our updated Stage 1/Stage 2 code imports this taxonomy from `model_contract.py`. Existing model weights and comparison adapters are not certified as implementing it; the category model requires new training.

## Categories and order

Use the following stable order when implementing category-based outputs and saving their label mappings:

1. `product_quality`
2. `functionality`
3. `performance`
4. `design`
5. `aesthetics`
6. `sensory_experience`
7. `value`
8. `packaging`
9. `seller_service`
10. `accuracy_of_description`

### product_quality

Overall build, material quality, durability, or craftsmanship.

### functionality

Whether the product has the expected features or capabilities.

### performance

How well the product performs its functions, including speed, responsiveness, lag, accuracy, sound, battery performance, results or effectiveness, or reliability.

### design

Ergonomics, physical form, layout, fit, comfort, feel in the hand, texture of non-beauty products, or ease of use.

### aesthetics

Purely visual appeal, including color, appearance, style, or look.

### sensory_experience

For beauty, skincare, and personal care products only (e.g., cleansers, soaps, moisturizers, perfumes): smell, fragrance, texture, consistency, and how the product feels on the skin (e.g., "mabango", "walang amoy", "smooth ang texture", "creamy", "foamy", "liquid").

### value

Whether the product is worth its price, including sulit, affordable, expensive, or overpriced.

### packaging

How the product was packaged, including box condition, wrapping, protection, presentation, or whether the item was securely packed.

### seller_service

Concrete seller-controlled behavior such as responsiveness, communication, helpfulness, problem resolution, customer support, or assistance. Does not include general thanks or praise for the seller.

### accuracy_of_description

Whether the received product matches what was advertised, as stated by the reviewer, including size, color, variation, specifications, or stated features (e.g., "same as in the picture", "wrong size", "hindi pareho sa description").

## Annotation boundaries

- A review may evaluate multiple categories, each with its own Positive, Neutral, or Negative sentiment. Category detection is therefore a multi-label task.
- Distinguish the existence of a capability (`functionality`) from how well it works (`performance`). For example, Bluetooth availability concerns functionality; Bluetooth lag concerns performance.
- Distinguish material/build durability (`product_quality`) from operating results (`performance`). Sound and battery performance should not be automatically collapsed into product quality.
- Physical comfort and usability belong to `design`; purely visual appeal belongs to `aesthetics`.
- Product context determines whether a texture or feel statement belongs to `sensory_experience` or `design`. Do not infer sensory eligibility from words such as "smooth" alone.
- A stated listing comparison belongs to `accuracy_of_description`; a standalone color preference belongs to `aesthetics`, and a standalone fit observation belongs to `design`.
- Packaging protection and box condition belong to `packaging`. Seller communication and problem resolution belong to `seller_service`. General seller thanks or praise is insufficient for seller-service annotation.
- There is no `delivery` category in this taxonomy. Delivery speed or courier behavior alone must not be automatically mapped to `seller_service` or `packaging`.
- Mentioning a category or a neutral descriptive property does not by itself establish an evaluative opinion. Apply the finalized opinion/experience annotation rules before assigning polarity or including a mention in sentiment aggregation.
- Aspect annotations and review-quality labels are separate decisions. Being outside the aspect taxonomy does not by itself determine whether a review is Authentic, Deceptive, LIV, or Irrelevant.

## Implementation implications

- A category-based ABSA detector needs ten outputs corresponding to the order above, with independent category probabilities. Its sentiment component needs a polarity prediction conditioned on the target category.
- Training, inference, comparison adapters, evaluation, and displayed results must use the same category identifiers and definitions.
- Preserve annotation evidence separately from the category identifier; an extracted phrase is not itself a fixed category prediction.
- Persist taxonomy version/order and selected thresholds with newly trained artifacts. Existing phrase-based weights must not be relabeled as a ten-category model merely by changing configuration metadata.
- Validation and final testing should evaluate category detection and per-category polarity separately. For paired sentiment-model comparisons, provide the same ground-truth target category to both models.

## Existing implementation status when adopted

The original `stage2/absa_model.py` was a BIO phrase extractor. It has now been replaced with ten-category detection and category-conditioned sentiment. The new code preserves old weights and saves newly trained artifacts separately. `backend/services/taxonomy.py` still contains the old six-category keyword projection; comparison work is intentionally deferred.

Training and evaluation use one adjudicated polarity per review/category. Repeated evidence with the same category and polarity is merged. Opposing polarities within the same review/category must be resolved by the annotation protocol before training; they are not silently averaged.
