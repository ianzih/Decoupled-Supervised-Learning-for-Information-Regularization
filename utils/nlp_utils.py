import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import re
import string
import itertools
import numpy as np
from collections import Counter
from tqdm import tqdm

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
        
        
def data_cleansing(_text, _labels, doRemove=False):
    """
    Detect or remove the empty samples.
    """
    assert len(_text)==len(_labels), "Text and label list need to be the same length."
    
    if doRemove == False:
        print("Warning: same samples in data preprocessing outputs may have empty list. This will damage the model.")
        return _text, _labels

    clear_text = []
    clear_label = []

    for idx , word in enumerate(_text):
        if word.strip():
            clear_text.append(word)
            clear_label.append(_labels[idx])
    
    if len(_text) >= len(clear_text) and (doRemove == True):
        print("Info: Detect the empty samples, and remove them!")
        print("Size change: {0}->{1}".format(len(_text), len(clear_text)))
    
    return clear_text, clear_label


def data_preprocessing(text, remove_stopword=False):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    if remove_stopword:
        text = [word for word in text.split() if word not in stop_words]
    else:
        text = [word for word in text.split()]
    text = ' '.join(text)

    return text


def create_vocab(corpus, vocab_size=30000):
    # corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()

    if vocab_size > len(sorted_words):
        v = len(sorted_words)
    else:
        v = vocab_size - 2

    vocab_to_int = {w: i + 2 for i, (w, c) in enumerate(sorted_words[:v])}

    vocab_to_int['<pad>'] = 0
    vocab_to_int['<unk>'] = 1
    print('vocab size', len(vocab_to_int))

    return vocab_to_int


def get_word_vector(vocab, emb='glove'):
    if emb == 'glove':
        fname = 'glove.6B.300d.txt'
        dim = 300

        with open(fname, 'rt', encoding='utf8') as fi:
            full_content = fi.read().strip().split('\n')

        data = {}
        for i in tqdm(range(len(full_content)), total=len(full_content), desc='loading glove vocabs...'):
            i_word = full_content[i].split(' ')[0]
            if i_word not in vocab.keys():
                continue
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            data[i_word] = i_embeddings
    else:
        raise Exception('emb not implemented')
    
    torch_vocab = [torch.as_tensor(data.get(word, torch.rand(dim))) for word in vocab.keys()]
    # torch_vocab = []
    # for word in vocab.keys():
    #     try:
    #         torch_vocab.append(torch.tensor(data[word]))
    #         find += 1
    #     except:
    #         torch_vocab.append(torch.rand(300))

    return torch.stack(torch_vocab, dim=0)


def tokenize(sentence): 
    return nltk.word_tokenize(sentence)


def add_noise(loader, class_num, noise_rate):
    """ Referenced from https://github.com/PaulAlbert31/LabelNoiseCorrection """
    print("[DATA INFO] Use noise rate {} in training dataset.".format(float(noise_rate)))
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.y]
    text = [sample_i for sample_i in loader.sampler.data_source.x]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_rate*100)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(set(range(class_num)))
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    # loader.sampler.data_source.x = text
    loader.sampler.data_source.y = noisy_labels

    return noisy_labels


class Textset(Dataset):
    def __init__(self, text, label, vocab, max_len, pad_value=0, pad_token='<pad>'):
        super().__init__()
        self.pad_value = pad_value
        self.pad_token = pad_token

        method = 1
        self.handle(text, label, vocab, max_len, method)

    def handle(self, text, label, vocab, max_len, method=1):

        if method == 0:
            print("[Textset] Using method 0")
            new_text = []
            for t in text:
                t_split = t.split(' ')
                if len(t_split) > max_len:
                    t_split = t_split[:max_len]
                    new_text.append(' '.join(t_split))
                else:
                    while len(t_split) < max_len:
                        t_split.append(self.pad_token)
                    new_text.append(' '.join(t_split))
            self.x = new_text
            self.y = label
            self.vocab = vocab
        
        elif method == 1:
            print("[Textset] Using method 1")
            new_text = []
            for t in text:
                t_split = t.split(' ')
                if len(t_split) > max_len:
                    t_split = t_split[:max_len]
                    new_text.append(' '.join(t_split))
                else:
                    new_text.append(' '.join(t_split))
            self.x = new_text
            self.y = label
            self.vocab = vocab

        elif method == 2:
            print("[Textset] Using method 2")
            new_text = []
            for t in text:
                if len(t) > max_len:
                    t = t[:max_len]
                    new_text.append(t)
                else:
                    new_text.append(t)
            self.x = new_text
            self.y = label
            self.vocab = vocab
        else:
            raise RuntimeError("Textset method setting error!")

    def collate(self, batch):

        x = [torch.tensor(x) for x, y in batch]
        y = [torch.tensor(y) for x, y in batch]
        x_tensor = pad_sequence(x, True)
        y = torch.tensor(y)
        return x_tensor, y

    def convert2id(self, text):
        r = []
        for word in text.split():
            if word in self.vocab.keys():
                r.append(self.vocab[word])
            else:
                r.append(self.vocab['<unk>'])
        return r

    def __getitem__(self, idx):
        text = self.x[idx]
        word_id = self.convert2id(text)
        return word_id, self.y[idx]

    def __len__(self):
        return len(self.x)