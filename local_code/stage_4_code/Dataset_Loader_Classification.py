'''
Dataset loader for IMDB movie review sentiment classification
Loads train/test splits, cleans text, builds vocab, encodes sequences
'''

import os
import re
from collections import Counter
from local_code.base_class.dataset import dataset

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'as', 'into', 'through',
    'during', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'shall', 'can', 'i', 'me', 'my', 'we', 'our', 'you',
    'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
    'their', 'this', 'that', 'these', 'those', 'not', 'no', 'so', 'if',
    'then', 'than', 'too', 'very', 'just', 'also', 'more', 'most', 'other',
    'some', 'such', 'only', 'same', 'here', 'there', 'when', 'where', 'all',
    'both', 'each', 'few', 'own', 'what', 'which', 'who', 'whom', 'how',
}


class Dataset_Loader_Classification(dataset):
    dataset_source_folder_path = None
    max_vocab_size = 20000
    max_seq_len = 300

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', ' ', text)   # strip HTML tags like <br />
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
        tokens = text.split()
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    def _load_split(self, split):
        texts, labels = [], []
        for label_name, label_val in [('pos', 1), ('neg', 0)]:
            folder = os.path.join(self.dataset_source_folder_path, split, label_name)
            fnames = sorted(f for f in os.listdir(folder) if f.endswith('.txt'))
            for fname in fnames:
                with open(os.path.join(folder, fname), 'r', encoding='utf-8', errors='replace') as f:
                    texts.append(self.clean_text(f.read()))
                labels.append(label_val)
        return texts, labels

    def _build_vocab(self, texts):
        counter = Counter()
        for tokens in texts:
            counter.update(tokens)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in counter.most_common(self.max_vocab_size):
            vocab[word] = len(vocab)
        return vocab

    def _encode(self, texts, vocab):
        encoded = []
        for tokens in texts:
            ids = [vocab.get(t, 1) for t in tokens]
            if len(ids) >= self.max_seq_len:
                ids = ids[:self.max_seq_len]
            else:
                ids += [0] * (self.max_seq_len - len(ids))
            encoded.append(ids)
        return encoded

    def load(self):
        print('Loading IMDB classification data...')
        train_texts, train_labels = self._load_split('train')
        test_texts, test_labels = self._load_split('test')

        print(f'  Train: {len(train_texts)} reviews  |  Test: {len(test_texts)} reviews')

        vocab = self._build_vocab(train_texts)
        print(f'  Vocabulary size: {len(vocab)}')

        train_X = self._encode(train_texts, vocab)
        test_X  = self._encode(test_texts, vocab)

        self.data = {
            'train': {'X': train_X, 'y': train_labels},
            'test':  {'X': test_X,  'y': test_labels},
            'vocab': vocab,
        }
        return self.data
