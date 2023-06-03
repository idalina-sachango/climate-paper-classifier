import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from transformers import BertTokenizer
import os

path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'preprocessed_data.csv'))
print(path)

# path="../../data/preprocessed_data.csv"
df = pd.read_csv(path)

seed = 30255    
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class Processing():
    def __init__(self):
        self.df = pd.DataFrame()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.vocab = pd.DataFrame()
        self.seq_len = 512
        self.num_words = 0

    def load(self):
        from sklearn import preprocessing
        self.df = pd.read_csv(path)
        self.df = self.df[['CLASS', 'SPACY_PREPROCESSED']]
        self.df = self.df.dropna()
        self.df['SPACY_PREPROCESSED'] = self.df['SPACY_PREPROCESSED'].str.replace(r'<[^<>]*>', '', regex=True) # drop HTML tags

        le = preprocessing.LabelEncoder()
        le.fit(self.df['CLASS'])
        self.df['LABEL'] = le.transform(self.df['CLASS'])

        self.df = self.df[['LABEL', 'SPACY_PREPROCESSED']]

        self.df['LABEL'].value_counts(dropna=False)

        self.text = self.df["SPACY_PREPROCESSED"].values
        self.target = self.df["LABEL"].values
    
    def tokenize_and_build_vocabulary(self):
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator

        tokenizer = get_tokenizer('basic_english')

        def yield_tokens(train_iter):
            for text in train_iter:
                yield tokenizer(text)

        # tokenize and build vocab over all words 
        train_iter = iter(self.text.tolist())
        self.tokenized = list(map(lambda text : tokenizer(text), train_iter))
        # re-initialize train iter
        train_iter = iter(self.text.tolist())
        # build vocab
        self.vocab = build_vocab_from_iterator(
            yield_tokens(train_iter), specials=["<unk>"], max_tokens = self.seq_len)
        self.vocab.set_default_index(self.vocab['<unk>'])
        # set num words in vocab
        self.num_words = len(self.vocab)

    def word_to_idx(self):
        # Index representation	
        self.index_representation = list()
        for sentence in self.tokenized:
            temp_sentence = list()
            for word in sentence:
                idx = self.vocab.lookup_indices([word])
                temp_sentence.extend(self.vocab.lookup_indices([word]))
            self.index_representation.append(temp_sentence)

    def padding_sentences(self):
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0

        pad_idx = 0
        self.padded = list()
        for sentence in self.index_representation: # tensors
            if len(sentence) < self.seq_len:
                while len(sentence) < self.seq_len: #max_length
                    sentence.append(pad_idx)
                sentence = torch.tensor(sentence)
                self.padded.append(sentence)
            else:
                sentence = torch.tensor(sentence[:self.seq_len])
                self.padded.append(sentence) # new code
        self.padded = torch.stack(self.padded)

    def split_data(self):
        self.target = torch.tensor(self.target)
        # Concatenating the Padded Vectors, Labels
        dataset = TensorDataset(self.padded, self.target)

        # compute train/validation/test split sizes
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # split dataset randomly into train/validation/test sets
        train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

        # create data loaders for each set
        self.train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=16, shuffle=True)


    def process_all(self):
        self.load()
        self.tokenize_and_build_vocabulary()
        self.word_to_idx()
        self.padding_sentences()
        self.split_data()

    def prints(self):
        # Task 1
        print(f"Task 1: The number of words in the Vocab object is {len(self.vocab)}.")

        # # Task 2
        stoi_dict = self.vocab.get_stoi()
        word = "energy"
        print(f"Task 2: The index of the word '{word}' is {stoi_dict[word]}.")

        # # Task 3
        itos_dict = self.vocab.get_itos()
        idx = 500
        print(f"Task 3: The word at index 500 is '{itos_dict[idx]}'.")

        # # Task 4:
        word = "<unk>"
        print(f"Task 4: The index of the word '{word}' is {stoi_dict[word]}. Resetting default index to this value.")
