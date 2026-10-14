"""Export the best checkpoint of a completed run; require saved dataset provenance."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_contract import INPUT_VERSION, ROBERTA_TARGET_COLUMN
from model_data import fingerprint, load_experiment
from training_augmentation import augmentation_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--augmentations')
    p.add_argument('--allow-unreviewed-augmentations', action='store_true')
    args = p.parse_args()
    frame = load_experiment(args.data, args.splits, args.augmentations,
                            allow_unreviewed=args.allow_unreviewed_augmentations)
    train = frame[frame.partition.eq('train')]
    val = frame[frame.partition.eq('validation')]
    output = Path(args.output).resolve()
    run = json.loads((output / 'run_metadata.json').read_text(encoding='utf-8'))
    if (run['input_version'] != INPUT_VERSION or run.get('target_column') != ROBERTA_TARGET_COLUMN
            or run['training_sha256'] != fingerprint(train)
            or run['validation_sha256'] != fingerprint(val)):
        raise ValueError('Dataset, partitions, or input contract differ from the saved run')
    steps = math.ceil(len(train) / run['batch_size']) * run['epochs']
    state = json.loads((output / 'checkpoints' / f'checkpoint-{steps}' / 'trainer_state.json').read_text(encoding='utf-8'))
    if state['global_step'] != steps or state['max_steps'] != steps or state['epoch'] != run['epochs']:
        raise ValueError('Training schedule is incomplete')
    best = Path(state['best_model_checkpoint']).resolve()
    if best.parent != output / 'checkpoints':
        raise ValueError('Best checkpoint is outside this run')
    best_step = int(best.name.removeprefix('checkpoint-'))
    metrics = next(x for x in reversed(state['log_history']) if x.get('step') == best_step and 'eval_macro_f1' in x)
    if metrics['eval_macro_f1'] != state['best_metric']:
        raise ValueError('Validation selection mismatch')
    from safetensors import safe_open
    with safe_open(str(best / 'model.safetensors'), framework='pt', device='cpu') as weights:
        if not list(weights.keys()):
            raise ValueError('Empty weights')
    for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json'):
        if (best / name).exists():
            shutil.copy2(best / name, output / name)
    metadata = {**run, 'fit_review_ids': train.review_id.tolist(),
        'validation_review_ids': val.review_id.tolist(), 'augmentation_usage': augmentation_summary(train),
        'text_input': 'product_title + product_description + review_text',
        'validation_metrics': {k: v for k, v in metrics.items() if k.startswith('eval_')},
        'recovered_from_checkpoint': str(best)}
    (output / 'training_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Recovered {best.name}; no training repeated.', flush=True)


if __name__ == '__main__':
    main()
