import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tokenizers import Tokenizer
from src.model import DecoderSLM

def compute_loss_and_metrics(model, x, y, pad_id, sen_id, fre_id, rot_id):
    """
    x: (batch, seq_len) input tokens
    y: (batch, seq_len) target tokens (shifted x)
    """
    # model returns (seq_len, vocab_size) if given 1D sequence. We map over batch
    logits, _ = jax.vmap(model)(x)
    
    # Calculate Cross Entropy Loss
    # labels: y
    vocab_size = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(y, vocab_size)
    
    # loss per token
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_losses = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    
    # mask out padding
    valid_mask = (y != pad_id)
    loss = jnp.sum(token_losses * valid_mask) / jnp.clip(jnp.sum(valid_mask), a_min=1)
    
    # perplexity
    perplexity = jnp.exp(loss)
    
    # Sentiment Accuracy: Look for [SEN] token in inputs
    # If x has [SEN] at index i, the prediction for sentiment is logits at i, target is y at i.
    is_sen_token = (x == sen_id)
    
    # We only care where SEN token is found
    sen_logits = jnp.where(is_sen_token[..., None], logits, -jnp.inf) # mask out non-SEN
    preds = jnp.argmax(sen_logits, axis=-1)
    
    # is the prediction correct where is_sen_token is True?
    correct = (preds == y) & is_sen_token
    total_sen = jnp.sum(is_sen_token)
    
    # For sequences without [SEN] due to truncation, ignore in accuracy
    accuracy = jnp.where(total_sen > 0, jnp.sum(correct) / total_sen, jnp.nan)
    
    return loss, (perplexity, accuracy)

@eqx.filter_value_and_grad(has_aux=True)
def compute_grads(model, x, y, pad_id, sen_id, fre_id, rot_id):
    return compute_loss_and_metrics(model, x, y, pad_id, sen_id, fre_id, rot_id)

def get_dataloader(x_path, batch_size, shuffle=True):
    X = np.load(x_path, mmap_mode='r')
    n_samples, seq_len = X.shape
    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
        
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_X = X[batch_idx]
        
        # Next token prediction: x is [:-1], y is [1:]
        inputs = batch_X[:, :-1]
        targets = batch_X[:, 1:]
        
        yield jnp.array(inputs), jnp.array(targets)

def train_slm(
    data_dir, 
    tokenizer_path, 
    max_epochs=50, 
    patience=None, 
    batch_size=64, 
    learning_rate=3e-4, 
    d_model=256, 
    num_heads=4, 
    num_layers=4, 
    mlp_hidden=1024, 
    max_steps=None
):
    """
    Trains the Decoder SLM on the pretokenized dataset.
    
    Returns:
        tuple: (trained_model, history_dict)
    """
    print("Loading tokenizer and finding special IDs...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    sen_id = tokenizer.token_to_id("[SEN]")
    fre_id = tokenizer.token_to_id("[FRE]")
    rot_id = tokenizer.token_to_id("[ROT]")
    
    # Infer max sequence length and dataset sizes from data
    X_train_shape = np.load(os.path.join(data_dir, "train_x.npy"), mmap_mode='r').shape
    X_val_shape = np.load(os.path.join(data_dir, "val_x.npy"), mmap_mode='r').shape
    max_seq_len = X_train_shape[1] - 1 # because we shift by 1 for autoregressive target
    
    total_train_batches = int(np.ceil(X_train_shape[0] / batch_size))
    total_val_batches = int(np.ceil(X_val_shape[0] / batch_size))
    
    print(f"Vocab size: {vocab_size}, Max Sequence Length: {max_seq_len}")
    
    key = jax.random.PRNGKey(42)
    model = DecoderSLM(
        vocab_size=vocab_size, 
        max_seq_len=max_seq_len, 
        d_model=d_model, 
        num_heads=num_heads, 
        num_layers=num_layers, 
        mlp_hidden=mlp_hidden, 
        key=key
    )
    
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def step(model, opt_state, x, y):
        (loss, (ppl, acc)), grads = compute_grads(model, x, y, pad_id, sen_id, fre_id, rot_id)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, ppl, acc

    @eqx.filter_jit
    def evaluate(model, x, y):
        return compute_loss_and_metrics(model, x, y, pad_id, sen_id, fre_id, rot_id)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    history = {
        "train_loss": [], "train_ppl": [], "train_acc": [],
        "val_loss": [], "val_ppl": [], "val_acc": []
    }
    
    print("\nStarting Training Loop...")
    
    for epoch in range(1, max_epochs + 1):
        # Training
        train_loss_total = 0.0
        train_ppl_total = 0.0
        train_acc_total = 0.0
        train_batches = 0
        train_acc_batches = 0
        
        train_gen = get_dataloader(os.path.join(data_dir, "train_x.npy"), batch_size, shuffle=True)
        
        total_steps = min(total_train_batches, max_steps) if max_steps is not None else total_train_batches
        with tqdm(total=total_steps, desc=f"Epoch {epoch:03d} [Train]", leave=False) as train_pbar:
            for x_batch, y_batch in train_gen:
                model, opt_state, loss, ppl, acc = step(model, opt_state, x_batch, y_batch)
                train_loss_total += loss
                train_ppl_total += ppl
                if not jnp.isnan(acc):
                    train_acc_total += acc
                    train_acc_batches += 1
                train_batches += 1
                    
                train_pbar.update(1)
                train_pbar.set_postfix({'loss': f"{train_loss_total / train_batches:.4f}"})
                
                if max_steps is not None and train_batches >= max_steps:
                    break
            
        avg_train_loss = float(train_loss_total / train_batches)
        avg_train_ppl = float(train_ppl_total / train_batches)
        avg_train_acc = float((train_acc_total / train_acc_batches) * 100 if train_acc_batches > 0 else 0.0)

        # Validation
        val_loss_total = 0.0
        val_ppl_total = 0.0
        val_acc_total = 0.0
        val_batches = 0
        val_acc_batches = 0
        
        val_gen = get_dataloader(os.path.join(data_dir, "val_x.npy"), batch_size, shuffle=False)
        with tqdm(total=total_val_batches, desc=f"Epoch {epoch:03d} [Val]", leave=False) as val_pbar:
            for x_batch, y_batch in val_gen:
                loss, (ppl, acc) = evaluate(model, x_batch, y_batch)
                val_loss_total += loss
                val_ppl_total += ppl
                if not jnp.isnan(acc):
                    val_acc_total += acc
                    val_acc_batches += 1
                val_batches += 1
                
                val_pbar.update(1)
                val_pbar.set_postfix({'loss': f"{val_loss_total / val_batches:.4f}"})
            
        avg_val_loss = float(val_loss_total / val_batches)
        avg_val_ppl = float(val_ppl_total / val_batches)
        avg_val_acc = float((val_acc_total / val_acc_batches) * 100 if val_acc_batches > 0 else 0.0)
        
        # Update History
        history["train_loss"].append(avg_train_loss)
        history["train_ppl"].append(avg_train_ppl)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_ppl"].append(avg_val_ppl)
        history["val_acc"].append(avg_val_acc)

        tqdm.write(f"Epoch {epoch:03d} | Train: L={avg_train_loss:.4f} Ppl={avg_train_ppl:.2f} Acc={avg_train_acc:.2f}% | Val: L={avg_val_loss:.4f} Ppl={avg_val_ppl:.2f} Acc={avg_val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            patience_counter = 0
            eqx.tree_serialise_leaves(os.path.join(data_dir, "best_model.eqx"), model)
        else:
            patience_counter += 1
            
        if patience is not None and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}! Reverting to best model.")
            model = best_model
            break
            
    print("Training Complete.")
    return model, history

def plot_training_history(history, save_path="data/training_plot.png"):
    """
    Plots the training and validation curves for Loss, Perplexity, and Accuracy.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss Plot
    axes[0].plot(epochs, history["train_loss"], label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], label='Val Loss')
    axes[0].set_title('Cross Entropy Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Perplexity Plot
    axes[1].plot(epochs, history["train_ppl"], label='Train Ppl')
    axes[1].plot(epochs, history["val_ppl"], label='Val Ppl')
    axes[1].set_title('Perplexity')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Perplexity')
    axes[1].legend()
    axes[1].grid(True)
    
    # Accuracy Plot
    axes[2].plot(epochs, history["train_acc"], label='Train Acc')
    axes[2].plot(epochs, history["val_acc"], label='Val Acc')
    axes[2].set_title('Sentiment Accuracy (%)')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")
    plt.show()
    plt.close()

def test_train():
    """
    Verifies the training loop and plotting capabilities.
    """
    data_dir = "data"
    tokenizer_path = "data/tokenizer.json"
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}. Please run tokenizer_dev.py first.")
        return

    print("--- Running Training Verification Test ---")
    model, history = train_slm(
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        max_epochs=10,
        max_steps=10,
        batch_size=64,
        d_model=128,
        num_heads=2,
        num_layers=2
    )

    plot_training_history(history)
    
    weights_path = os.path.join(data_dir, "best_model.eqx")
    if os.path.exists(weights_path):
        print(f"Success: Best model weights found at {weights_path}")
    else:
        print(f"Warning: Best model weights NOT found at {weights_path}")

if __name__ == "__main__":
    test_train()
