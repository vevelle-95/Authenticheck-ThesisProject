"""Isolated JSON bridge for the external Lu et al. inference environment.

The external checkout must provide an executable accepting one JSON request on
stdin and writing one JSON response to stdout. It runs in its own Python
environment; no baseline package is imported into this process.
"""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from .taxonomy import category_for, normalize_sentiment


class BaselineUnavailable(RuntimeError):
    pass


class BaselineExecutionError(RuntimeError):
    pass


def _split_segments(text):
    # This locates candidate evidence; it does not infer aspect sentiment.
    return [piece.strip() for piece in re.split(r"[.!?;\n]+|\s+\b(?:but|however|pero)\b\s+", text, flags=re.IGNORECASE) if piece.strip()]


def project_review(review, prediction):
    """Project coarse sentiment to matched text clauses with explicit provenance."""
    if not isinstance(prediction, dict):
        raise BaselineExecutionError("Baseline review result must be an object")
    try:
        holistic = normalize_sentiment(prediction["sentiment"])
    except (KeyError, ValueError) as error:
        raise BaselineExecutionError("Baseline must return a valid holistic sentiment for each review") from error
    segments = prediction.get("segments")
    if segments is None:
        segments = [{"text": text} for text in _split_segments(review["text"])]
    if not isinstance(segments, list):
        raise BaselineExecutionError("Baseline segments must be an array")
    aspects = []
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise BaselineExecutionError("Each baseline segment needs text")
        evidence = segment["text"].strip()
        if not evidence or evidence not in review["text"]:
            raise BaselineExecutionError("Baseline segment must occur in its review text")
        category = category_for(evidence)
        if category is None:
            continue
        has_segment_sentiment = "sentiment" in segment
        try:
            sentiment = normalize_sentiment(segment["sentiment"]) if has_segment_sentiment else holistic
        except ValueError as error:
            raise BaselineExecutionError("Invalid segment sentiment") from error
        aspects.append({
            "category": category, "evidence": evidence, "sentiment": sentiment,
            "sentiment_source": "baseline_segment" if has_segment_sentiment else "holistic_projection",
        })
    return {"id": review["id"], "holistic_sentiment": holistic, "aspects": aspects}


def run_baseline(reviews):
    root = Path(os.environ.get("LU_ET_AL_REPO", "/baselines/lu_et_al_repo")).expanduser()
    command = os.environ.get("LU_ET_AL_COMMAND", "").strip()
    python = os.environ.get("LU_ET_AL_PYTHON", "").strip()
    entrypoint = os.environ.get("LU_ET_AL_ENTRYPOINT", "").strip()
    if not root.is_dir():
        raise BaselineUnavailable("Configure LU_ET_AL_REPO for the external baseline")
    if command:
        # Windows CreateProcess parses a command line string without invoking a shell.
        argv = command if os.name == "nt" else shlex.split(command)
    elif python and entrypoint:
        argv = [python, str(Path(__file__).with_name("lu_et_al_worker.py"))]
    else:
        raise BaselineUnavailable("Configure LU_ET_AL_COMMAND or LU_ET_AL_PYTHON and LU_ET_AL_ENTRYPOINT")
    if not argv:
        raise BaselineUnavailable("LU_ET_AL_COMMAND is empty")
    payload = {"reviews": reviews}
    try:
        completed = subprocess.run(
            argv, input=json.dumps(payload), text=True, encoding="utf-8",
            capture_output=True, cwd=root, timeout=120, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BaselineExecutionError(f"Baseline process could not complete: {type(error).__name__}") from error
    if completed.returncode:
        raise BaselineExecutionError(f"Baseline process exited with code {completed.returncode}")
    try:
        output = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BaselineExecutionError("Baseline did not return valid JSON on stdout") from error
    predicted = output.get("reviews") if isinstance(output, dict) else None
    if not isinstance(predicted, list) or len(predicted) != len(reviews):
        raise BaselineExecutionError("Baseline response must contain one result per review")
    if any(not isinstance(row, dict) or row.get("id") != review["id"] for review, row in zip(reviews, predicted)):
        raise BaselineExecutionError("Baseline response review IDs must match request order")
    return [project_review(review, row) for review, row in zip(reviews, predicted)]
