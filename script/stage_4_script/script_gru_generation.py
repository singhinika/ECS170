'''
Stage 4-5 (GRU): Train a GRU language model on the jokes dataset, then generate
                 jokes from 3-word seeds.

Key architectural changes vs. base RNN (script_rnn_generation_edited.py):
  cell_type  : 'GRU'  — replaces vanilla tanh-RNN with Gated Recurrent Unit cells.
               GRU uses two gates (reset and update) instead of the LSTM's three,
               making it faster to train while still solving the vanishing gradient
               problem.  It tends to converge faster than LSTM on shorter sequences
               (most jokes are under 25 words), making it a strong candidate here.

  embed_dim  : 256 (was 128) — wider embeddings for richer token representation.

  hidden_dim : 512 (was 256) — more capacity for modelling joke structure.

  max_epoch  : 75 (vs 80 for LSTM) — GRU typically converges slightly faster than
               LSTM because it has fewer parameters, so fewer epochs are needed.

  learning_rate: 5e-4 (was 1e-3) — consistent with the LSTM script; prevents
               overshooting with the larger hidden dimension.

  dropout    : 0.35 (was 0.3) — moderate increase for the larger model; less than
               LSTM because GRU has fewer parameters and is already less prone to
               overfitting.

  temperature: 0.75 (was 0.8) — slightly lower than base RNN for more coherent text;
               slightly higher than LSTM to preserve GRU's natural diversity.

  top_k      : 12 (was 10) — modest expansion from RNN baseline; keeps output
               focused while allowing natural vocabulary variation.

Run from project root:
    python -m script.stage_4_script.script_gru_generation
'''

import os
import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation     import Method_RNN_Generation
from local_code.stage_4_code.plot_utils                import plot_generation_loss

MODEL_NAME = 'GRU_Generation'
RESULT_DIR = f'./result/stage_4_result/{MODEL_NAME}'

FIXED_SEEDS = [
    ['what', 'did', 'the'],
    ["don't", 'you', 'hate'],
    ['two', 'artists', 'had'],
    ['why', 'did', 'the'],
    ['what', 'do', 'you'],
    ['why', 'does', 'a'],
    ['how', 'do', 'you'],
]


def get_seeds(raw_jokes, vocab, n=7):
    seeds, seen = [], set()
    for s in FIXED_SEEDS:
        key = tuple(s)
        if key not in seen and all(w in vocab for w in s):
            seeds.append(s)
            seen.add(key)
        if len(seeds) >= n:
            break
    for joke in raw_jokes:
        if len(seeds) >= n:
            break
        if len(joke) >= 4:
            key = tuple(joke[:3])
            if key not in seen and all(w in vocab for w in joke[:3]):
                seeds.append(list(joke[:3]))
                seen.add(key)
    return seeds


def find_original(raw_jokes, seed):
    for joke in raw_jokes:
        if joke[:len(seed)] == seed:
            return ' '.join(w for w in joke if w != '<EOS>')
    return None


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ---- data ----
    data_obj = Dataset_Loader_Generation('jokes', 'Short Jokes Dataset')
    data_obj.dataset_source_file_path = './data/stage_4_data/text_generation/data'
    loaded_data = data_obj.load()

    # ---- model: GRU with improved hyperparameters ----
    method_obj = Method_RNN_Generation(MODEL_NAME, '')
    method_obj.data          = loaded_data
    method_obj.vocab_size    = len(loaded_data['vocab'])
    method_obj.cell_type     = 'GRU'       # key change: reset+update gates
    method_obj.embed_dim     = 256         # richer token representations
    method_obj.hidden_dim    = 512         # more modelling capacity
    method_obj.num_layers    = 2
    method_obj.dropout       = 0.35        # moderate regularisation
    method_obj.max_epoch     = 75          # slightly fewer than LSTM; GRU converges faster
    method_obj.learning_rate = 5e-4        # lower LR for stable convergence
    method_obj.batch_size    = 128

    # ---- train ----
    print('************ Start Training (GRU) ************')
    method_obj.run()
    print(f'  Final training loss: {method_obj.train_losses[-1]:.4f}')

    # ---- save loss curve ----
    plot_generation_loss(
        method_obj.train_losses,
        save_dir=RESULT_DIR,
        model_name=MODEL_NAME,
    )

    # ---- generate jokes ----
    vocab     = loaded_data['vocab']
    raw_jokes = loaded_data['raw_jokes']
    seeds     = get_seeds(raw_jokes, vocab, n=7)

    divider = '=' * 65
    output_lines = [
        'GRU Text Generation Results\n',
        f'Model: 2-layer GRU  |  embed=256  |  hidden=512  |  '
        f'Epochs: {method_obj.max_epoch}  '
        f'|  Final Loss: {method_obj.train_losses[-1]:.4f}\n',
        '\n',
        'Architecture changes vs. base RNN:\n',
        '  cell_type=GRU    : reset+update gates solve vanishing gradients with fewer params than LSTM\n',
        '  embed_dim=256    : richer word representations for puns/wordplay\n',
        '  hidden_dim=512   : more capacity to model joke structure\n',
        '  max_epoch=75     : faster convergence than LSTM due to fewer parameters\n',
        '  learning_rate=5e-4: stable convergence with more parameters\n',
        '  dropout=0.35     : moderate regularisation (less than LSTM, more than RNN)\n',
        '  temperature=0.75 : slightly sharper distribution than base RNN\n',
        '  top_k=12         : modest candidate expansion from RNN baseline\n\n',
    ]

    print('\n' + divider)
    print('GENERATED JOKES vs. ORIGINAL  (GRU)')
    print(divider)

    for i, seed in enumerate(seeds, 1):
        # temperature=0.75, top_k=12 balance coherence and variety
        generated = method_obj.generate(seed, max_len=60, temperature=0.75, top_k=12)
        original  = find_original(raw_jokes, seed)

        block = (
            f'[{i}] Seed : "{" ".join(seed)}"\n'
            f'    Generated : {generated}\n'
        )
        if original:
            block += f'    Original  : {original}\n'
        block += '\n'

        print(block, end='')
        output_lines.append(block)

    summary = (
        f'\nTraining Loss per Epoch:\n'
        + '\n'.join(f'  Epoch {e+1:>2}: {l:.4f}'
                    for e, l in enumerate(method_obj.train_losses))
        + '\n'
    )
    output_lines.append(summary)

    out_path = os.path.join(RESULT_DIR, 'generated_jokes.txt')
    with open(out_path, 'w') as f:
        f.writelines(output_lines)

    print(divider)
    print(f'Results saved to: {out_path}')
    print(f'Loss plot saved to: {RESULT_DIR}/{MODEL_NAME}_loss.png')
    print('************ Finish ************')
