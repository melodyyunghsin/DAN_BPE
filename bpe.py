from collections import Counter
from sentiment_data import read_sentiment_examples
import os
from utils import Indexer

def load_or_learn_merges(infile, num_merges, cache_file="bpe_merges.txt"):
    """Reuse saved merges if there are enough; otherwise learn them and save."""
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            merges = [tuple(line.split()) for line in f]
        if len(merges) >= num_merges:
            return merges[:num_merges]
    print(f"Learning {num_merges} BPE merges")
    merges = learn_bpe(infile, num_merges)
    with open(cache_file, "w") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    return merges

def build_subword_indexer(train_file, tokenizer):
    indexer = Indexer()
    indexer.add_and_get_index("PAD")   # index 0
    indexer.add_and_get_index("UNK")   # index 1
    for ex in read_sentiment_examples(train_file):
        # tokenize ex.words then add every piece to the indexer
        for piece in tokenizer.tokenize(ex.words):
            indexer.add_and_get_index(piece)
    return indexer

def get_word_freqs(infile):
    """Count each word, then write it as a tuple of characters + '</w>'."""
    counts = Counter()
    for ex in read_sentiment_examples(infile):
        counts.update(ex.words)
    return {tuple(word) + ("</w>",): freq for word, freq in counts.items()}


def count_pairs(word_freqs):
    """Return a Counter: (symbol1, symbol2) -> how often that pair appears."""
    pairs = Counter()
    for symbols, freq in word_freqs.items():
        # loop over every pair (symbols[i], symbols[i+1]) and add freq to pairs[pair]
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_symbols(symbols, pair):
    """Glue every adjacent (a, b) inside one list of symbols. Returns a new list."""
    a, b = pair
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            # add the merged symbol to out, go to i + 2
            out.append(a + b)
            i += 2
        else:
            # add symbols[i] to out, go to i + 1
            out.append(symbols[i])
            i += 1
    return out

def merge_pair(pair, word_freqs):
    return {tuple(merge_symbols(symbols, pair)): freq for symbols, freq in word_freqs.items()}


def learn_bpe_from_freqs(word_freqs, num_merges):
    """Learn num_merges merges. Returns the merges as a list, in order."""
    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(word_freqs)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0] # the most frequent pair
        word_freqs = merge_pair(best, word_freqs)
        merges.append(best)
    return merges


def learn_bpe(infile, num_merges):
    return learn_bpe_from_freqs(get_word_freqs(infile), num_merges)

class BPETokenizer:
    def __init__(self, merges):
        # rank: pair -> its position in the merge list (smaller = learned earlier)
        self.rank = {pair: i for i, pair in enumerate(merges)}
        self.cache = {}   # remembers words already split, so each is only done once

    def tokenize_word(self, word):
        if word in self.cache:
            return self.cache[word]
        symbols = list(word) + ["</w>"]
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            known = [pair for pair in pairs if pair in self.rank]
            if not known:
                break
            best = min(known, key=lambda p: self.rank[p])
            symbols = merge_symbols(symbols, best)
        self.cache[word] = symbols
        return symbols

    def tokenize(self, words):
        """Split a whole sentence (list of words) into one flat list of pieces."""
        pieces = []
        for w in words:
            pieces.extend(self.tokenize_word(w))
        return pieces

if __name__ == "__main__":
    toy = {
        ("c", "a", "t", "</w>"): 4,
        ("c", "a", "t", "s", "</w>"): 2,
        ("h", "a", "t", "</w>"): 3,
    }
    merges = learn_bpe_from_freqs(toy, 3)
    print(merges)

    tok = BPETokenizer(merges)
    print(tok.tokenize_word("chat"))
    print(tok.tokenize_word("hats"))
    print(tok.tokenize_word("cat"))
    print(tok.tokenize(["cat", "chat"]))