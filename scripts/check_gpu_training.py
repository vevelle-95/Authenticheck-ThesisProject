"""Tiny real GPU fitting/save checks, isolated from experiment artifacts."""
import argparse
import gc
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('quality', 'absa'))
    p.add_argument('--gpu-memory-gb', type=float, default=3)
    args = p.parse_args()
    if not args.phase:
        for phase in ('quality', 'absa'):
            subprocess.run([sys.executable, '-u', __file__, '--phase', phase,
                            '--gpu-memory-gb', str(args.gpu_memory_gb)], check=True)
        print('Both GPU smoke checks passed.', flush=True)
        return
    import torch
    from gpu_memory import configure_gpu_memory
    from model_data import load_experiment
    configure_gpu_memory(args.gpu_memory_gb)
    torch.set_num_threads(2)
    frame = load_experiment('data/compiled_v1/cleaned.csv', 'data/compiled_v1/splits.json', require_annotations=True)
    output = Path('tmp/gpu_training_check_v3') / args.phase
    if args.phase == 'quality':
        from stage1.train_roberta import fit_roberta, predict_probabilities
        tiny = frame[frame.partition.eq('train')].groupby('label', group_keys=False).head(1)
        model, tokenizer = fit_roberta(tiny, None, output, epochs=1, batch_size=1,
                                      max_length=128, low_memory=True)
        gc.collect()
        predict_probabilities(model, tokenizer, tiny, batch_size=1)
    else:
        from stage2.fine_tune_absa import train_absa
        authentic = frame[frame.partition.eq('train') & frame.label.eq(0)]
        indices = []
        for polarity in range(3):
            indices.append(next(i for i, row in authentic.iterrows()
                                if any(a['sentiment'] == polarity for a in row.annotations)))
        tiny = authentic.loc[list(dict.fromkeys(indices))].copy()
        tiny['partition'] = 'train'
        validation = tiny.copy()
        validation['partition'] = 'validation'
        import pandas as pd
        train_absa(pd.concat([tiny, validation]), output, epochs=1, batch_size=1,
                   max_length=128, low_memory=True)
    print(f"{args.phase} passed; peak PyTorch reserved={torch.cuda.max_memory_reserved() / 1e9:.2f} GB", flush=True)


if __name__ == '__main__':
    main()
