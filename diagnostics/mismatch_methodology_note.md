# Sentiment–star mismatch proposal

Status: unevaluated; CPU cross-fit stopped before its first fold completed. No metrics or candidate model exist; adviser/panel approval is required.

**Proposed feature.** For each predicted aspect, normalize Stage 2's three sigmoid polarity scores to sum to one; calculate `s_text = P(Positive) + 0.5 × P(Neutral)` and average across predicted-present aspects. Use `s_text = 0.5` if none are detected. Define mismatch as `abs(s_text − R_star)`. This keeps the score in `[0,1]` but is unvalidated. Chapter 2 may cite sentiment–rating inconsistency (Soneria & Jethva, 2025); verify the full reference and findings first.

**Proposed Chapter 3 revisions, subject to approval:**
- **Online Review definition:** “An eligible review contains text and a valid star rating; an image is optional.”
- **Sampling/Data Collection:** “Retain eligible reviews without usable images, record image availability, and encode missing-image `S_clip` as 0.”
- **Scope:** “Reviews without usable images remain in scope; image-derived evidence is unavailable for those records.”
- **M-CLIP/feature vector:** “If 7D wins validation, append mismatch to the six features; if 6D wins, replace `S_clip` with mismatch.” Do not select before validation.
