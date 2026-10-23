import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentiment_data import read_sentiment_examples, read_word_embeddings
import torch.nn.functional as F

def words_to_ids(words, indexer):
    ids = []
    for word in words:
        idx = indexer.index_of(word)
        if idx == -1:
            ids.append(1) # Assuming 1 is the index for unknown words
        else:
            ids.append(idx)
    return ids


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, indexer, tokenizer=None):
        examples = read_sentiment_examples(infile)

        # make a list of id_lists per sentence
        # id_lists = [words_to_ids(example.words, indexer) for example in examples]
        if tokenizer is not None:
            id_lists = [words_to_ids(tokenizer.tokenize(example.words), indexer) for example in examples]
        else:
            id_lists = [words_to_ids(example.words, indexer) for example in examples]
        
        max_len = max(len(lst) for lst in id_lists) # the length of the longest id_list

        # pad each id_list with 0 so every one has length max_len
        padded = [lst + [0] * (max_len - len(lst)) for lst in id_lists]

        labels = [example.label for example in examples] # label list

        self.ids = torch.tensor(padded, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


class DAN(nn.Module): 
    def __init__(self, embedding_layer, hidden_size=100, dropout=0.3):
        super().__init__()
        # self.embedding = embeddings.get_initialized_embedding_layer(frozen=frozen)
        emb_dim = embedding_layer.embedding_dim
        self.embedding = embedding_layer
        self.dropout = nn.Dropout(dropout)
        # fc1: emb_dim numbers -> hidden_size numbers
        self.fc1 = nn.Linear(emb_dim, hidden_size)
        # fc2: hidden_size numbers -> 2 scores
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        emb = self.embedding(x) #[batch size, sentence length, embedding size]
        mask = (x != 0).float() #[batch size, sentence length]
        emb = emb * mask.unsqueeze(2)

        # sum emb over the word dimension -> [batch size, embedding size]
        summed = emb.sum(dim=1)
        # number of real words per sentence -> [batch size, 1]
        lengths = mask.sum(dim=1, keepdim=True)
        avg = self.dropout(summed / lengths)

        # fc1, ReLU, fc2, log_softmax
        hidden = self.dropout(F.relu(self.fc1(avg)))
        output = self.fc2(hidden)
        out = self.log_softmax(output)
        return out

if __name__ == "__main__":
    embs = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    train_data = SentimentDatasetDAN("data/train.txt", embs.word_indexer)
    print("num examples:", len(train_data))
    print("first example:", train_data[0])

    loader = DataLoader(train_data, batch_size=16, shuffle=True)
    X, y = next(iter(loader))
    print("batch shapes:", X.shape, y.shape)
    
    model = DAN(embs.get_initialized_embedding_layer(frozen=False), hidden_size=100)
    out = model(X)
    print("output shape:", out.shape)
    print("probabilities of first sentence:", out[0].exp())