"""One-time frozen CLIP extraction; defaults to training/validation Authentic reviews."""

import argparse
import json

from datasets.authenticheck_data import load_experiment, load_reviews
from datasets.clip_cache import ClipFeatureCache
from runtime import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--csv')
    parser.add_argument('--splits')
    parser.add_argument('--augmentations', help='Use the same approved augmentation CSV as training')
    parser.add_argument('--allow-unreviewed-augmentations', action='store_true', help='Include pending drafts for an experimental training run')
    parser.add_argument('--partition', nargs='+', choices=['train', 'validation', 'test', 'all'], default=['train', 'validation'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    augmentation_path = args.augmentations or config['data'].get('augmentations')
    if 'all' in args.partition:
        if args.allow_unreviewed_augmentations:
            raise ValueError('Unreviewed augmentation mode requires named partitions and an augmentation CSV')
        if augmentation_path:
            raise ValueError('--partition all is for arbitrary inference CSVs; use named partitions with augmentations')
        frame = load_reviews(resolve_path(args.csv or config['data']['csv']))
    else:
        frame = load_experiment(
            resolve_path(args.csv or config['data']['csv']),
            resolve_path(args.splits or config['data']['splits']),
            resolve_path(augmentation_path) if augmentation_path else None,
            allow_unreviewed=args.allow_unreviewed_augmentations,
        )
        frame = frame[frame.partition.isin(args.partition)]
        # Test extraction prepares every review for unfiltered prediction.
        frame = frame[frame.partition.eq('test') | frame.label.eq(0)]
    print(json.dumps(ClipFeatureCache(config).prepare(frame, force=args.force), indent=2))


if __name__ == '__main__':
    main()
