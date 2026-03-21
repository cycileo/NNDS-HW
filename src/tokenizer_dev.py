import pandas as pd
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

SPECIAL_TOKENS = ["[PAD]", "[SOS]", "[EOS]", "[SOR]", "[RP]", "[SEN]", "[FRE]", "[ROT]", "[UNK]"]

def train_tokenizer(reviews, vocab_size, show_progress=False):
    """
    Trains a BPE tokenizer on a list of strings given a target vocabulary size.
    Returns the trained tokenizers.Tokenizer object.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, show_progress=show_progress)
    
    tokenizer.train_from_iterator(reviews, trainer)
    return tokenizer

def analyze_vocab_sizes(data_path, vocab_sizes):
    """
    Trains tokenizers across multiple vocab sizes and prints length statistics in a unified table.
    """
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    reviews = df['formatted_review'].tolist()
    
    # Calculate baseline word statistics
    word_lengths = [len(r.split()) for r in reviews]
    avg_words = np.mean(word_lengths)
    
    print(f"Average words per review: {avg_words:.2f} (Max: {np.max(word_lengths)}, Min: {np.min(word_lengths)})")
    
    stats = []
    
    # Take a sample review to compare tokenizations side-by-side
    sample_review = reviews[0]
    sample_tokens_by_vocab = {}
    
    # Progress bar across vocab sizes
    for size in tqdm(vocab_sizes, desc="Analyzing Vocab Configs"):
        tokenizer = train_tokenizer(reviews, size, show_progress=False)
        
        encodings = tokenizer.encode_batch(reviews)
        lengths = [len(e.ids) for e in encodings]
        
        avg_len = np.mean(lengths)
        frag_ratio = avg_len / avg_words
        
        stats.append({
            "Vocab Size": size,
            "Avg Tokens": f"{avg_len:.2f}",
            "Max Tokens": np.max(lengths),
            "Min Tokens": np.min(lengths),
            "Tokens/Word": f"{frag_ratio:.2f}"
        })
        
        sample_tokens_by_vocab[size] = tokenizer.encode(sample_review).tokens

    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'Vocab Size':<12} | {'Avg Tokens':<12} | {'Max Tokens':<12} | {'Min Tokens':<12} | {'Tokens/Word'}")
    print("-" * 80)
    for s in stats:
        print(f"{s['Vocab Size']:<12} | {s['Avg Tokens']:<12} | {s['Max Tokens']:<12} | {s['Min Tokens']:<12} | {s['Tokens/Word']}")
    print("="*80)
    
    # Print Sample Comparisons
    print(f"\nSample Review Comparison (Original: {len(sample_review.split())} words)")
    print("-" * 80)
    max_len = 150 
    
    for size in vocab_sizes:
        tokens = sample_tokens_by_vocab[size]
        token_str = " | ".join(tokens)
        if len(token_str) > max_len:
            token_str = token_str[:max_len] + "..."
        print(f"Vocab {size:<6} ->  {token_str}")

def pretokenize_dataset(data_path, output_dir, vocab_size=10000, max_seq_len=256, splits=(0.90, 0.05, 0.05)):
    """
    Trains BPE tokenizer, encodes dataset, pads to max_seq_len, 
    splits into train/val/test, and saves to .npy files.
    """
    print(f"Loading dataset from: {data_path}...")
    df = pd.read_csv(data_path)
    reviews = df['formatted_review'].tolist()
    labels = df['is_fresh'].values.astype(bool)
    
    print(f"Training BPE Tokenizer (Vocab Size: {vocab_size})...")
    tokenizer = train_tokenizer(reviews, vocab_size, show_progress=True)
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    pad_id = tokenizer.token_to_id("[PAD]")
    
    print("Encoding reviews...")
    encodings = tokenizer.encode_batch(reviews)
    
    valid_sequences = []
    valid_labels = []
    dropped_count = 0
    total_count = len(encodings)
    
    for i, enc in enumerate(encodings):
        seq = enc.ids
        if len(seq) > max_seq_len:
            dropped_count += 1
        else:
            valid_sequences.append(seq)
            valid_labels.append(labels[i])
            
    if dropped_count > 0:
        drop_pct = (dropped_count / total_count) * 100
        print(f"Dropped {dropped_count}/{total_count} ({drop_pct:.2f}%) sequences exceeding {max_seq_len} tokens.")
    
    n_valid = len(valid_sequences)
    X = np.full((n_valid, max_seq_len), pad_id, dtype=np.int32)
    y = np.array(valid_labels, dtype=bool)
    
    for i, seq in enumerate(valid_sequences):
        X[i, :len(seq)] = seq
        
    print(f"Dataset split: Train {splits[0]*100}% / Val {splits[1]*100}% / Test {splits[2]*100}%")
    np.random.seed(42)  
    indices = np.random.permutation(n_valid)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    train_end = int(splits[0] * n_valid)
    val_end = train_end + int(splits[1] * n_valid)
    
    X_train, y_train = X_shuffled[:train_end], y_shuffled[:train_end]
    X_val, y_val = X_shuffled[train_end:val_end], y_shuffled[train_end:val_end]
    X_test, y_test = X_shuffled[val_end:], y_shuffled[val_end:]
    
    np.save(os.path.join(output_dir, "train_x.npy"), X_train)
    np.save(os.path.join(output_dir, "train_y.npy"), y_train)
    np.save(os.path.join(output_dir, "val_x.npy"), X_val)
    np.save(os.path.join(output_dir, "val_y.npy"), y_val)
    np.save(os.path.join(output_dir, "test_x.npy"), X_test)
    np.save(os.path.join(output_dir, "test_y.npy"), y_test)
    
    print(f"Done! Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer Developer Script")
    parser.add_argument("--data_path", type=str, default="data/processed_reviews.csv")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--analyze", action="store_true", help="Run vocabulary size analysis natively without exporting.")
    parser.add_argument("--vocab_sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000], help="List of vocab sizes to analyze.")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_vocab_sizes(args.data_path, args.vocab_sizes)
    else:
        pretokenize_dataset(
            data_path=args.data_path,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len
        )
