# CSE 256 PA1: DAN, BPE, Skip-Gram

## Setup

Requires Python 3 with `torch`, `numpy`, `matplotlib`, and `scikit-learn`.
Run every command from this folder (the code reads `data/` with relative paths).

## Files

| File | Contents |
|---|---|
| `main.py` | training/eval loops, command line parse, plots |
| `DANmodels.py` | Part 1: `SentimentDatasetDAN` (sentences to padded ID tensors) and the `DAN` model |
| `bpe.py` | Part 2: Byte Pair Encoding: learning merges, `BPETokenizer`, subword vocabulary |
| `BOWmodels.py`, `sentiment_data.py`, `utils.py` |  |

## Running each part

```bash
python main.py --model BOW          # provided bag-of-words baseline
python main.py --model DAN          # Part 1a: DAN with GloVe embeddings
python main.py --model DAN --random_init   # Part 1b: randomly initialized embeddings
python main.py --model SUBWORDDAN   # Part 2: DAN on BPE subwords
```

The defaults are the best setting found: 50d GloVe (fine-tuned), dropout 0.3,
hidden size 100, Adam with lr 0.001, 20 epochs, batch size 16.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--emb` | 50 | Embedding size. For `DAN` this selects the GloVe file (50 or 300). For random/subword embeddings it is the vector length. |
| `--dropout` | 0.3 | Dropout probability, applied to the averaged embedding and to the hidden layer |
| `--frozen` | off | Keep the GloVe embeddings fixed during training (`DAN` only) |
| `--random_init` | off | Use randomly initialized embeddings instead of GloVe (`DAN` only) |
| `--hidden` | 100 | Hidden layer size |
| `--lr` | 0.001 | Adam learning rate |
| `--epochs` | 20 | Training epochs |
| `--merges` | 2000 | Number of BPE merges (`SUBWORDDAN` only) |

Examples:

```bash
python main.py --model DAN --emb 300 --frozen
python main.py --model SUBWORDDAN --merges 5000
```

## Notes

- **BPE merge cache.** BPE is learned on `data/train.txt` only. Learned merges are saved to
  `bpe_merges.txt`. Because BPE is greedy, the first N merges are the same no matter how many
  are learned, so a run with N merges reuses the first N lines of the cache. If the cache has
  fewer than N merges, all N are relearned (about 19 s per 500 merges; 10,000 merges takes a few minutes).
- **Subword embeddings** are always randomly initialized, since GloVe has no vectors for subword pieces.
- **Plots.** Each run saves a train/dev accuracy plot, named after its settings
  (e.g. `dan_50d_drop0.3.png`, `subword_5000.png`).
- Results vary by about ±1% between runs (random initialization and shuffling).

## AI assistance
Claude was used to polish some of the wording of the reports and formatting the file. All codes are written by myself with some AI assistance on checking the structure of the code.
