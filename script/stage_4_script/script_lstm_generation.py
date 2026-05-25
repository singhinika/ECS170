'''
Stage 4-5 (LSTM): Train an LSTM language model on the jokes dataset, then generate
                  jokes from 3-word seeds.

Key architectural changes vs. base RNN (script_rnn_generation_edited.py):
  cell_type  : 'LSTM'  — replaces vanilla tanh-RNN with Long Short-Term Memory cells.
               LSTM adds an explicit memory cell (c_t) and three learnable gates
               (input, forget, output) that let the network decide what to remember
               across many timesteps.  For joke generation this means the model can
               remember the joke "setup" at timestep 3 while writing the "punchline"
               at timestep 20, producing more coherent jokes.

  embed_dim  : 256 (was 128) — wider word embeddings give a richer representation
               of each token, especially helpful for the puns and wordplay in jokes.

  hidden_dim : 512 (was 256) — more hidden units let the LSTM model more complex
               linguistic patterns and multi-step reasoning in joke structure.

  max_epoch  : 80 (was 50) — LSTM has more parameters, so more passes are needed to
               converge.  The loss should drop further than the RNN baseline.

  learning_rate: 5e-4 (was 1e-3) — lower LR prevents the optimizer from overshooting
               the sharper loss landscape that comes with more parameters.

  dropout    : 0.4 (was 0.3) — slightly stronger regularisation to offset the higher
               model capacity and avoid memorising individual jokes.

  temperature: 0.7 (was 0.8) — lower temperature makes the next-word distribution
               sharper, reducing the chance of random nonsense words.

  top_k      : 15 (was 10) — slightly wider candidate set than the RNN baseline to
               preserve creativity while still filtering low-probability tokens.

Run from project root:
    python -m script.stage_4_script.script_lstm_generation
'''

import os
import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation     import Method_RNN_Generation
from local_code.stage_4_code.plot_utils                import plot_generation_loss

MODEL_NAME = 'LSTM_Generation'
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

    # ---- model: LSTM with improved hyperparameters ----
    method_obj = Method_RNN_Generation(MODEL_NAME, '')
    method_obj.data          = loaded_data
    method_obj.vocab_size    = len(loaded_data['vocab'])
    method_obj.cell_type     = 'LSTM'      # key change: gated memory cells
    method_obj.embed_dim     = 256         # richer token representations
    method_obj.hidden_dim    = 512         # more modelling capacity
    method_obj.num_layers    = 2
    method_obj.dropout       = 0.4         # stronger regularisation for larger model
    method_obj.max_epoch     = 80          # more passes to converge
    method_obj.learning_rate = 5e-4        # lower LR for stable convergence
    method_obj.batch_size    = 128

    # ---- train ----
    print('************ Start Training (LSTM) ************')
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
        'LSTM Text Generation Results\n',
        f'Model: 2-layer LSTM  |  embed=256  |  hidden=512  |  '
        f'Epochs: {method_obj.max_epoch}  '
        f'|  Final Loss: {method_obj.train_losses[-1]:.4f}\n',
        '\n',
        'Architecture changes vs. base RNN:\n',
        '  cell_type=LSTM   : gated memory prevents vanishing gradients over long sequences\n',
        '  embed_dim=256    : richer word representations for puns/wordplay\n',
        '  hidden_dim=512   : more capacity to model joke structure\n',
        '  max_epoch=80     : more training for the larger model\n',
        '  learning_rate=5e-4: stable convergence with more parameters\n',
        '  dropout=0.4      : prevents memorising individual jokes\n',
        '  temperature=0.7  : sharper distribution for more coherent outputs\n',
        '  top_k=15         : slightly wider candidate vocabulary\n\n',
    ]

    print('\n' + divider)
    print('GENERATED JOKES vs. ORIGINAL  (LSTM)')
    print(divider)

    for i, seed in enumerate(seeds, 1):
        # temperature=0.7 and top_k=15 produce more coherent completions
        generated = method_obj.generate(seed, max_len=60, temperature=0.7, top_k=15)
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
