'''
Dataset loader for joke text generation
Loads jokes CSV, tokenizes, builds vocab, creates full-sequence teacher-forcing pairs
'''

import csv
import re
from collections import Counter
from local_code.base_class.dataset import dataset

PAD = '<PAD>'   # index 0 — also used as ignore_index in loss
UNK = '<UNK>'   # index 1
EOS = '<EOS>'   # index 2  — appended to every joke


class Dataset_Loader_Generation(dataset):
    dataset_source_file_path = None
    max_vocab_size = 8000
    max_joke_len   = 50    # jokes longer than this are truncated

    def clean_joke(self, text):
        text = text.lower()
        text = re.sub(r'([?.!,])', r' \1 ', text)   # treat punctuation as separate tokens
        text = re.sub(r"[^a-z0-9?.!,'\s]", ' ', text)
        return [t for t in text.split() if t]

    def load(self):
        print('Loading jokes generation data...')
        raw_jokes = []
        with open(self.dataset_source_file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tokens = self.clean_joke(row['Joke'])
                if tokens:
                    raw_jokes.append(tokens)

        print(f'  Loaded {len(raw_jokes)} jokes')

        # Build vocab from all jokes (we train and generate on the same set)
        counter = Counter()
        for joke in raw_jokes:
            counter.update(joke)

        vocab = {PAD: 0, UNK: 1, EOS: 2}
        for word, _ in counter.most_common(self.max_vocab_size):
            if word not in vocab:
                vocab[word] = len(vocab)
        id2word = {v: k for k, v in vocab.items()}
        print(f'  Vocabulary size: {len(vocab)}')

        # Encode jokes and add EOS; truncate to max_joke_len
        encoded_jokes = []
        for joke in raw_jokes:
            ids = [vocab.get(t, 1) for t in joke] + [vocab[EOS]]
            if len(ids) > self.max_joke_len + 1:
                ids = ids[:self.max_joke_len] + [vocab[EOS]]
            encoded_jokes.append(ids)

        # Build full-sequence teacher-forcing pairs
        # input  = ids[:-1],  target = ids[1:]
        # both padded to (max_joke_len) with PAD=0
        max_len = self.max_joke_len
        train_X, train_y = [], []
        for ids in encoded_jokes:
            inp  = ids[:-1]
            tgt  = ids[1:]
            pad_len = max_len - len(inp)
            train_X.append(inp + [0] * pad_len)
            train_y.append(tgt + [0] * pad_len)

        print(f'  Training sequences: {len(train_X)}  (length {max_len})')

        self.data = {
            'train':  {'X': train_X, 'y': train_y},
            'vocab':   vocab,
            'id2word': id2word,
            'raw_jokes': raw_jokes,       # original tokenized jokes for evaluation
        }
        return self.data
