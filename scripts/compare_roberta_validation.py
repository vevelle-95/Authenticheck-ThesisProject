"""Compare two saved quality encoders on the existing validation products only."""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_contract import CLASS_NAMES, DEFAULT_DATA_PATH, DEFAULT_SPLITS_PATH, INPUT_VERSION, ROBERTA_TARGET_COLUMN
from model_data import fingerprint, load_experiment
from model_metrics import classification_metrics
from stage1.train_roberta import predict_probabilities


def validate_encoder(directory, assigned):
    metadata = json.loads((directory / 'training_metadata.json').read_text(encoding='utf-8'))
    train = assigned[assigned.partition.eq('train')]
    validation = assigned[assigned.partition.eq('validation')]
    if metadata.get('input_version') != INPUT_VERSION or metadata.get('target_column') != ROBERTA_TARGET_COLUMN:
        raise ValueError(f'{directory}: incompatible text input or training target')
    if (set(metadata.get('fit_review_ids', [])) != set(train.review_id)
            or set(metadata.get('validation_review_ids', [])) != set(validation.review_id)
            or metadata.get('training_sha256') != fingerprint(train)):
        raise ValueError(f'{directory}: dataset contents or split membership changed')
    run_path = directory / 'run_metadata.json'
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding='utf-8'))
        if run.get('validation_sha256') != fingerprint(validation):
            raise ValueError(f'{directory}: validation contents changed')
    return metadata


def metrics_with_classes(targets, predictions):
    from sklearn.metrics import precision_recall_fscore_support

    result = classification_metrics(targets, predictions, range(4))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=list(range(4)), zero_division=0)
    result['per_class'] = {
        name: {'precision': float(precision[i]), 'recall': float(recall[i]),
               'f1': float(f1[i]), 'support': int(support[i])}
        for i, name in enumerate(CLASS_NAMES)
    }
    return result


def evaluate_encoder(directory, validation, batch_size):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f'Evaluating {directory} on {len(validation)} validation reviews', flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True)
    if model.config.input_contract_version != INPUT_VERSION or model.config.quality_target_column != ROBERTA_TARGET_COLUMN:
        raise ValueError(f'{directory}: saved weights use an incompatible input/target contract')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    probabilities = predict_probabilities(model, tokenizer, validation, batch_size=batch_size)
    predictions = probabilities.argmax(axis=1)
    result = {
        'against_text_label': metrics_with_classes(validation.text_target, predictions),
        'against_ground_truth': metrics_with_classes(validation.label, predictions),
    }
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, predictions, probabilities


def checkpoint_selection(directory):
    states = sorted((directory / 'checkpoints').glob('checkpoint-*/trainer_state.json'),
                    key=lambda path: int(path.parent.name.split('-')[-1]))
    if not states:
        return None
    state = json.loads(states[-1].read_text(encoding='utf-8'))
    best = state.get('best_model_checkpoint')
    best_step = int(best.replace('\\', '/').split('/')[-1].split('-')[-1]) if best else None
    selected = next((row for row in state['log_history']
                     if row.get('step') == best_step and 'eval_macro_f1' in row), {})
    return {'best_epoch': selected.get('epoch'), 'best_macro_f1': state.get('best_metric'),
            'checkpoint': best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(DEFAULT_DATA_PATH))
    parser.add_argument('--splits', default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--reference-augmentations', help='Only reconstructs reference provenance; candidate remains real-only')
    parser.add_argument('--allow-unreviewed-reference-augmentations', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', default=16, type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise ValueError('Use a new comparison output path to preserve previous results')
    if args.output.suffix != '.json':
        parser.error('--output must be a .json file')

    original = load_experiment(args.data, args.splits)
    reference_assigned = load_experiment(
        args.data, args.splits, args.reference_augmentations,
        allow_unreviewed=args.allow_unreviewed_reference_augmentations)
    candidate_metadata = validate_encoder(args.candidate, original)
    reference_metadata = validate_encoder(args.reference, reference_assigned)
    if candidate_metadata.get('augmentation_usage', {}).get('approved_rows', 0) or candidate_metadata.get('augmentation_usage', {}).get('unreviewed_rows', 0):
        raise ValueError('Candidate must have no augmented training rows')
    if candidate_metadata['max_length'] != reference_metadata['max_length']:
        raise ValueError('Encoder token limits differ; this is not an augmentation-only comparison')
    validation = original[original.partition.eq('validation')].copy()
    reference, ref_pred, ref_prob = evaluate_encoder(args.reference, validation, args.batch_size)
    candidate, cand_pred, cand_prob = evaluate_encoder(args.candidate, validation, args.batch_size)
    delta = {
        target: {metric: candidate[target][metric] - reference[target][metric]
                 for metric in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')}
        for target in ('against_text_label', 'against_ground_truth')
    }
    records = [
        {'review_id': row.review_id, 'product_id': row.product_id,
         'product_title': row.product_title, 'product_description': row.product_description,
         'review_text': row.review_text, 'text_label': CLASS_NAMES[row.text_target],
         'ground_truth': CLASS_NAMES[row.label],
         'reference_prediction': CLASS_NAMES[int(ref_pred[i])],
         'candidate_prediction': CLASS_NAMES[int(cand_pred[i])],
         'reference_probabilities': dict(zip(CLASS_NAMES, ref_prob[i].tolist())),
         'candidate_probabilities': dict(zip(CLASS_NAMES, cand_prob[i].tolist()))}
        for i, row in enumerate(validation.itertuples())
    ]
    report = {
        'partition': 'validation', 'review_count': len(validation),
        'product_count': int(validation.product_id.nunique()), 'class_order': list(CLASS_NAMES),
        'validation_sha256': fingerprint(validation), 'test_predictions_generated': False,
        'selection_target': 'text_label', 'candidate_promoted': False,
        'reference_directory': str(args.reference), 'candidate_directory': str(args.candidate),
        'reference': reference, 'candidate': candidate, 'candidate_minus_reference': delta,
        'training_rows': {'reference': len(reference_metadata['fit_review_ids']),
                          'candidate': len(candidate_metadata['fit_review_ids'])},
        'training_text_label_counts': {
            name: {label: int(frame.loc[frame.partition.eq('train'), 'text_target'].eq(i).sum())
                   for i, label in enumerate(CLASS_NAMES)}
            for name, frame in [('reference', reference_assigned), ('candidate', original)]
        },
        'checkpoint_selection': {'reference': checkpoint_selection(args.reference),
                                 'candidate': checkpoint_selection(args.candidate)},
        'limitations': [
            'One seed and four validation products; this does not establish statistical significance.',
            'Historical reference resumed CPU training on GPU; device history is not matched.',
            'This compares quality encoders only; XGBoost, images, ABSA and test results are not reevaluated.',
        ],
        'records': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Real-only RoBERTa validation comparison', '',
             f'{len(validation)} original reviews from {validation.product_id.nunique()} validation products. No test predictions.', '',
             '| Target | Metric | Reference | Real-only | Change (percentage points) |',
             '| --- | --- | ---: | ---: | ---: |']
    for target in delta:
        for metric in ('accuracy', 'macro_f1'):
            lines.append(f'| {target} | {metric} | {reference[target][metric]*100:.2f}% | {candidate[target][metric]*100:.2f}% | {delta[target][metric]*100:+.2f} |')
    for target in delta:
        lines.extend(['', f'## {target}: per-class results', '',
                      '| Class | Support | Reference precision / recall / F1 | Real-only precision / recall / F1 |',
                      '| --- | ---: | --- | --- |'])
        for name in CLASS_NAMES:
            ref = reference[target]['per_class'][name]
            cand = candidate[target]['per_class'][name]
            values = lambda metrics: ' / '.join(f'{metrics[k]*100:.2f}%' for k in ('precision', 'recall', 'f1'))
            lines.append(f'| {name} | {ref["support"]} | {values(ref)} | {values(cand)} |')
    lines.extend(['', 'The model is selected by validation macro-F1 against text_label. Ground-truth metrics are reported separately.',
                  'This candidate is not deployed. A full pipeline comparison requires new compatible OOF features and XGBoost training.', '',
                  *report['limitations'], ''])
    args.output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('reference', 'candidate', 'candidate_minus_reference')}, indent=2), flush=True)
    print(f'Saved {args.output} and {args.output.with_suffix(".md")}', flush=True)


if __name__ == '__main__':
    main()
