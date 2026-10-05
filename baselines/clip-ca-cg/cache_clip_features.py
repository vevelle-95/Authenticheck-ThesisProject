"""One-time frozen CLIP extraction; defaults to training/validation Authentic reviews."""

import argparse
import json

from datasets.authenticheck_data import load_reviews, read_splits
from datasets.clip_cache import ClipFeatureCache
from runtime import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--csv')
    parser.add_argument('--splits')
    parser.add_argument('--partition', nargs='+', choices=['train', 'validation', 'test', 'all'], default=['train', 'validation'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_reviews(resolve_path(args.csv or config['data']['csv']), require_annotations='all' not in args.partition)
    if 'all' not in args.partition:
        frame = read_splits(frame, resolve_path(args.splits or config['data']['splits']))
        frame = frame[frame.partition.isin(args.partition)]
        # Test extraction prepares every review for unfiltered prediction.
        frame = frame[frame.partition.eq('test') | frame.label.eq(0)]
    print(json.dumps(ClipFeatureCache(config).prepare(frame, force=args.force), indent=2))


if __name__ == '__main__':
    main()
