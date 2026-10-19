"""Assemble the real-only quality models with an unchanged, cohort-matched ABSA."""

import argparse
import hashlib
import json
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_contract import (
    ASPECTS, INPUT_VERSION, MODEL_VERSION, QUALITY_TEXT_POLICY, ROBERTA_TARGET_COLUMN,
    SENSORY_POLICY, SENTIMENT_TARGET_POLICY, TAXONOMY_VERSION,
)
from model_data import fingerprint, load_experiment
from scripts.compare_roberta_validation import validate_encoder
from stage2.evaluate_models import check_training_provenance
from stage2.online_inference import OnlineInference


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('models/real_only_v1'))
    parser.add_argument('--source', type=Path, default=Path('models/quality_balanced_v5'))
    parser.add_argument('--allow-unreviewed-reference-augmentations', action='store_true')
    args = parser.parse_args()
    bundle, source = args.bundle.resolve(), args.source.resolve()
    if not bundle.is_relative_to(ROOT) or not source.is_relative_to(ROOT) or source == bundle:
        raise ValueError('Use distinct source and destination bundles inside this repository')
    target = bundle / 'absa_model'
    published_report = ROOT / 'reports/real_only_v1/validation_comparison.json'
    if target.exists() or (bundle / 'manifest.json').exists() or published_report.exists():
        raise FileExistsError('Preserve the existing assembled bundle and published report')
    original = load_experiment(ROOT / 'data/test_reviews.csv', ROOT / 'data/splits_dataset.json', require_annotations=True)
    reference = load_experiment(
        ROOT / 'data/test_reviews.csv', ROOT / 'data/splits_dataset.json', ROOT / 'data/augmented_reviews.csv',
        require_annotations=True, allow_unreviewed=args.allow_unreviewed_reference_augmentations)
    validate_encoder(bundle / 'dost_roberta', original)
    check_training_provenance(reference, source)
    source_metadata_path = source / 'absa_model/training_metadata.json'
    historical_metadata = json.loads(source_metadata_path.read_text(encoding='utf-8'))
    cohorts = {}
    for partition, id_key in [('train', 'fit_review_ids'), ('validation', 'validation_review_ids')]:
        old = reference[reference.partition.eq(partition) & reference.label.eq(0)].sort_values('review_id')
        new = original[original.partition.eq(partition) & original.label.eq(0)].sort_values('review_id')
        shared = sorted((set(new.columns) & set(old.columns)) - {'annotations'})
        if (not old[shared].reset_index(drop=True).equals(new[shared].reset_index(drop=True))
                or old.annotations.tolist() != new.annotations.tolist()
                or set(historical_metadata[id_key]) != set(new.review_id)):
            raise ValueError(f'Cannot reuse ABSA: {partition} Authentic inputs or annotations differ')
        cohorts[partition] = {'count': len(new), 'sha256': fingerprint(new),
                              'source_sha256': fingerprint(old)}
    report_source = ROOT / 'reports/generated/real_only_v1/stage1_validation_comparison.json'
    report = json.loads(report_source.read_text(encoding='utf-8'))
    classifier_meta = json.loads((bundle / 'xgboost_meta_classifier.metadata.json').read_text(encoding='utf-8'))
    if (report['partition'] != 'validation' or report['test_predictions_generated']
            or report['candidate_xgboost']['macro_f1'] != classifier_meta['validation_metrics']['macro_f1']):
        raise ValueError('Candidate validation report does not match the saved classifier')

    absa_files = ('absa_config.json', 'model.pt', 'encoder/config.json',
                  'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json')
    for name in absa_files:
        if not (source / 'absa_model' / name).is_file():
            raise FileNotFoundError(name)
    for name in absa_files:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / 'absa_model' / name, destination)
        if sha256(destination) != sha256(source / 'absa_model' / name):
            raise ValueError(f'ABSA copy checksum mismatch: {name}')
    shutil.copy2(source_metadata_path, target / 'source_training_metadata.json')
    metadata = dict(historical_metadata)
    metadata.update(
        dataset_sha256=fingerprint(original), training_sha256=cohorts['train']['sha256'],
        validation_sha256=cohorts['validation']['sha256'],
        checkpoint_reuse={
            'retrained': False, 'source_bundle': source.relative_to(ROOT).as_posix(),
            'source_dataset_sha256': historical_metadata['dataset_sha256'],
            'source_training_metadata_file': 'source_training_metadata.json',
            'source_training_metadata_sha256': sha256(source_metadata_path),
            'source_weights_sha256': sha256(source / 'absa_model/model.pt'),
            'verified_authentic_cohorts': cohorts,
            'verification': 'Exact equality of original input fields, parsed annotations and membership for both Authentic cohorts.',
            'dataset_scope': 'Assembly dataset fingerprints above refer to the original-only experiment; historical training metadata is preserved verbatim separately.',
        })
    (target / 'training_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    check_training_provenance(original, bundle)
    OnlineInference(bundle / 'dost_roberta', bundle / 'xgboost_meta_classifier.json', target).check_artifacts()
    published_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_source, published_report)
    runtime_files = ['xgboost_meta_classifier.json', 'xgboost_meta_classifier.metadata.json']
    runtime_files += [f'dost_roberta/{name}' for name in (
        'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'training_metadata.json')]
    runtime_files += [f'absa_model/{name}' for name in (*absa_files, 'training_metadata.json', 'source_training_metadata.json')]
    manifest = {
        'model_version': MODEL_VERSION, 'bundle_name': 'real_only_v1',
        'bundle_status': 'experimental_candidate', 'input_version': INPUT_VERSION,
        'taxonomy_version': TAXONOMY_VERSION, 'sensory_domain_policy': SENSORY_POLICY,
        'sentiment_target_policy': SENTIMENT_TARGET_POLICY, 'aspects': list(ASPECTS),
        'dataset_sha256': fingerprint(original), 'splits': 'data/splits_dataset.json',
        'augmentations': None, 'augmentation_policy': 'original-reviews-only',
        'augmentation_usage': {'approved_rows': 0, 'unreviewed_rows': 0},
        'quality_target_columns': {'roberta': ROBERTA_TARGET_COLUMN, 'xgboost': 'ground_truth'},
        'quality_text_input': 'review_text paired with bounded product_title + product_description',
        'quality_text_policy': QUALITY_TEXT_POLICY, 'quality_weighting': 'uniform',
        'xgboost_sample_weighting': 'row-balanced', 'oof_epochs': 6,
        'absa_text_input': 'review_text', 'absa_reused_from': source.relative_to(ROOT).as_posix(),
        'absa_retrained': False, 'test_evaluated': False,
        'evaluation_reports': [published_report.relative_to(ROOT).as_posix()],
        'validation_scope': 'Quality classifier only; API smoke checks do not measure full-pipeline accuracy.',
        'files_sha256': {name: sha256(bundle / name) for name in runtime_files},
        'environment': {name: version(name) for name in (
            'torch', 'transformers', 'datasets', 'scikit-learn', 'xgboost', 'sentence-transformers')},
    }
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Assembled and verified {len(runtime_files)} runtime files at {bundle}', flush=True)
    print('ABSA weights unchanged; source training metadata preserved. No retraining or test evaluation.', flush=True)


if __name__ == '__main__':
    main()
