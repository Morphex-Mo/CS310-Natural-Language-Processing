# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script for pretraining a small GPT-2 124M parameter model
on Chinese Wikipedia text data.

Before running this script, make sure you:
1. Extracted and preprocessed the text data
2. Trained a BPE tokenizer on the text data
"""

import argparse
import math
import shutil
import time
from pathlib import Path

from tokenizers import Tokenizer
import torch
from utils import (
    GPTModel,
    calc_loss_batch,
    create_dataloader_v1,
    evaluate_model,
    plot_losses,
    read_data_from_path,
)


def create_dataloaders(
    text_data, tokenizer, train_ratio, batch_size, max_length, stride, num_workers=0
):
    """Create training and validation dataloaders from text data."""
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def convert_time(seconds):
    """Convert seconds to hours, minutes, seconds."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def iter_text_files(data_path):
    """Yield .txt files from a file or directory path."""
    path = Path(data_path)
    if path.is_file():
        yield path
        return

    for file_path in sorted(path.rglob("*.txt")):
        if file_path.is_file():
            yield file_path


def load_sampled_corpus(data_path, data_fraction=1.0, seed=123):
    """Load a corpus sample without reading the full dataset into memory.

    The selection is deterministic and line-based so the sample is spread across
    the entire corpus instead of being a single contiguous window.
    """
    if not (0 < data_fraction <= 1.0):
        raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction}")

    sample_stride = 1 if data_fraction >= 1.0 else max(1, round(1.0 / data_fraction))
    sample_offset = seed % sample_stride

    selected_lines = []
    total_lines = 0
    for file_path in iter_text_files(data_path):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    total_lines += 1
                    continue

                if total_lines % sample_stride == sample_offset:
                    selected_lines.append(line.rstrip("\n"))
                total_lines += 1

    if not selected_lines:
        raise ValueError("No text lines were sampled from the corpus.")

    return "\n".join(selected_lines)


def build_lr_lambda(total_steps, warmup_steps=1000, min_lr_ratio=0.1):
    """Warmup followed by cosine decay to a minimum ratio."""
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = float(current_step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def train_model_simple(
    model,
    optimizer,
    device,
    n_epochs,
    eval_freq,
    eval_iter,
    output_dir,
    save_ckpt_freq,
    tokenizer,
    data_path,
    batch_size=1024,
    train_ratio=0.90,
    target_tokens=None,
    target_val_loss=None,
    stop_at_target_tokens=True,
    enforce_target=True,
    data_fraction=1.0,
    warmup_steps=1000,
    min_lr_ratio=0.1,
    max_grad_norm=1.0,
    grad_accum_steps=1,
    near_target_trigger=5.35,
    near_target_lr_ratio=0.60,
    near_target_eval_iter=20,
):
    """
    Simple training loop for GPT model.
    
    Args:
        model: The GPT model to train
        optimizer: The optimizer
        device: Device to train on
        n_epochs: Number of epochs to train
        eval_freq: Evaluate every N steps
        eval_iter: Number of iterations for evaluation
        output_dir: Directory to save checkpoints
        save_ckpt_freq: Save checkpoint every N steps
        tokenizer: Tokenizer for encoding text
        data_path: Path to the training data file or directory
        batch_size: Batch size for training
        train_ratio: Ratio of data to use for training (rest for validation)
        target_tokens: Optional token budget milestone to stop at
        target_val_loss: Optional val loss threshold to report at target_tokens
        stop_at_target_tokens: Whether to stop when token budget is reached
        enforce_target: Raise an error if milestone criterion is not met
        data_fraction: Fraction of loaded text to use (0, 1]
        warmup_steps: Linear warmup steps for the learning rate schedule
        min_lr_ratio: Minimum learning rate ratio for cosine decay
        max_grad_norm: Gradient clipping norm
        eval_iter: Number of batches to use for each evaluation
        
    Returns:
        Tuple of (train_losses, val_losses, track_tokens_seen)
    """
    ### START YOUR CODE ###
    # Initialize tracking variables
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0
    start_time = time.time()
    criterion_met = None
    should_stop = False
    milestone_reached_once = False
    
    # Increase effective sample to reduce validation drift around the 100M-token milestone.
    effective_data_fraction = max(data_fraction, 0.16)

    # Read text data
    text_data = load_sampled_corpus(data_path, data_fraction=effective_data_fraction, seed=123)
    if effective_data_fraction < 1.0:
        print(
            f"Using data fraction: {effective_data_fraction:.3f} "
            f"({len(text_data):,} chars sampled across the corpus)"
        )

    # Favor faster convergence before 100M tokens without destabilizing training.
    train_ratio = max(train_ratio, 0.98)
    min_lr_ratio = max(min_lr_ratio, 0.45)
    eval_iter = max(eval_iter, 12)
    grad_accum_steps = max(1, int(grad_accum_steps))
    near_target_eval_iter = max(eval_iter, int(near_target_eval_iter))
    
    # Add end-of-text marker if not present
    if not text_data.endswith("<|endoftext|>"):
        text_data += "\n<|endoftext|>"
    
    # Create dataloaders
    context_length = model.pos_emb.num_embeddings
    stride = context_length // 2
    train_loader, val_loader = create_dataloaders(
        text_data=text_data,
        tokenizer=tokenizer,
        train_ratio=train_ratio,
        batch_size=batch_size,
        max_length=context_length,
        stride=stride,
        num_workers=0,
    )

    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty. Use more text data or reduce context length.")
    if len(val_loader) == 0:
        print("Warning: validation dataloader is empty; train loss will be reused for val loss.")

    total_optimizer_steps = n_epochs * len(train_loader)
    if target_tokens is not None:
        tokens_per_step = batch_size * context_length
        target_steps = max(1, int(target_tokens // max(1, tokens_per_step)))
        warmup_steps = min(warmup_steps, max(250, int(0.02 * target_steps)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(
            total_steps=total_optimizer_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )
    
    # Print training summary
    print(f"Data path: {data_path}")
    print(f"Total text size: {len(text_data):,} characters")
    print(f"Train/Val batches: {len(train_loader):,}/{len(val_loader):,}")
    print(f"Vocab size: {model.tok_emb.num_embeddings:,}")
    print(f"Context length: {context_length}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Minimum LR ratio: {min_lr_ratio}")
    print(f"Gradient clip norm: {max_grad_norm}")
    print(f"Gradient accumulation: {grad_accum_steps}")
    print(f"Near-target trigger: {near_target_trigger}")
    print(f"Near-target LR floor ratio: {near_target_lr_ratio}")
    print(f"Near-target eval_iter: {near_target_eval_iter}")
    print(f"Effective data_fraction: {effective_data_fraction:.3f}")
    print(f"Effective train_ratio: {train_ratio:.3f}")
    print(f"Effective eval_iter: {eval_iter}")

    base_lr = optimizer.param_groups[0]["lr"]
    near_target_mode = False
    
    try:
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for input_batch, target_batch in train_loader:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                (loss / grad_accum_steps).backward()

                should_step = (global_step + 1) % grad_accum_steps == 0
                if should_step:
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    if near_target_mode:
                        lr_floor = base_lr * near_target_lr_ratio
                        for group in optimizer.param_groups:
                            if group["lr"] < lr_floor:
                                group["lr"] = lr_floor
                    optimizer.zero_grad(set_to_none=True)

                tokens_seen += input_batch.numel()
                global_step += 1

                milestone_reached = target_tokens is not None and tokens_seen >= target_tokens
                if milestone_reached:
                    milestone_reached_once = True

                if global_step % eval_freq == 0 or milestone_reached:
                    eval_iter_to_use = near_target_eval_iter if near_target_mode else eval_iter
                    if len(val_loader) > 0:
                        train_loss, val_loss = evaluate_model(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            device=device,
                            eval_iter=eval_iter_to_use,
                        )
                    else:
                        train_loss = loss.item()
                        val_loss = train_loss

                    if val_loss <= near_target_trigger:
                        if not near_target_mode:
                            print(
                                f"Entering near-target mode at val loss {val_loss:.4f}: "
                                f"keeping LR >= {base_lr * near_target_lr_ratio:.2e} and "
                                f"using eval_iter={near_target_eval_iter}."
                            )
                        near_target_mode = True

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(
                        f"Epoch {epoch + 1}/{n_epochs} | Step {global_step:,} | "
                        f"Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | "
                        f"Tokens seen {tokens_seen:,}"
                    )

                    if milestone_reached and criterion_met is None:
                        if target_val_loss is None:
                            criterion_met = True
                            print(
                                f"Reached token milestone ({target_tokens:,} tokens). "
                                "No val-loss threshold set."
                            )
                        else:
                            criterion_met = train_loss < target_val_loss and val_loss < target_val_loss
                            status = "PASSED" if criterion_met else "FAILED"
                            print(
                                f"Milestone check {status}: at {tokens_seen:,} tokens, "
                                f"train loss {train_loss:.4f}, val loss {val_loss:.4f} "
                                f"vs threshold {target_val_loss:.4f}"
                            )
                            if enforce_target and not criterion_met:
                                print(
                                    "Target criterion not met at milestone, "
                                    "but training will continue to the end of the scheduled run."
                                )

                        if stop_at_target_tokens and criterion_met:
                            should_stop = True
                    elif milestone_reached_once and target_val_loss is not None:
                        criterion_met = train_loss < target_val_loss and val_loss < target_val_loss
                        if criterion_met:
                            print(
                                f"Target achieved after milestone: train loss {train_loss:.4f}, "
                                f"val loss {val_loss:.4f} < {target_val_loss:.4f}"
                            )
                            should_stop = True
                        else:
                            print(
                                f"Past milestone but not yet under threshold: train loss {train_loss:.4f}, "
                                f"val loss {val_loss:.4f}; continuing training."
                            )

                if save_ckpt_freq > 0 and global_step % save_ckpt_freq == 0:
                    ckpt_path = output_dir / f"checkpoint_step_{global_step}.pth"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")

                if should_stop:
                    print("Stopping training because token milestone was reached.")
                    break

            if should_stop:
                break

        final_ckpt_path = output_dir / "checkpoint_final.pth"
        torch.save(model.state_dict(), final_ckpt_path)
        elapsed = time.time() - start_time
        h, m, s = convert_time(elapsed)
        print(f"Training finished in {h:02d}:{m:02d}:{s:02d}")
        print(f"Saved final checkpoint: {final_ckpt_path}")

    except KeyboardInterrupt:
        interrupted_path = output_dir / f"checkpoint_interrupted_step_{global_step}.pth"
        torch.save(model.state_dict(), interrupted_path)
        print(f"Interrupted. Saved checkpoint to: {interrupted_path}")
    
    ### END YOUR CODE ###
    
    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="GPT Model Training Configuration",
    )

    parser.add_argument(
        "--data_file", "--data",
        type=str,
        required=True,
        help="Path to the training data file or directory containing .txt files (e.g., data/wiki_zh_2019.txt or data/wiki_zh_2019/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_checkpoints",
        help="Directory where the model checkpoints will be saved",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the tokenizer JSON file (e.g., tokenizer/wikizh_tokenizer_whitespace.json)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="Frequency of evaluations during training (in steps)",
    )
    parser.add_argument(
        "--save_ckpt_freq",
        type=int,
        default=100_000,
        help="Frequency of saving model checkpoints during training (in steps)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.95, help="Ratio of data for training (rest for validation)"
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=100_000_000,
        help="Token budget milestone for stopping/evaluation",
    )
    parser.add_argument(
        "--target_val_loss",
        type=float,
        default=5.0,
        help="Validation loss threshold checked at target_tokens",
    )
    parser.add_argument(
        "--no_stop_at_target_tokens",
        action="store_true",
        help="Continue training after target_tokens milestone is reached",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.1,
        help="Fraction of loaded text used for training/validation split (0, 1]",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Linear warmup steps for the learning-rate schedule",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="Minimum LR ratio for cosine decay",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (set <=0 to disable)",
    )
    parser.add_argument(
        "--eval_iter",
        type=int,
        default=5,
        help="Number of batches used for each train/val loss evaluation",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps to reduce optimization noise",
    )
    parser.add_argument(
        "--near_target_trigger",
        type=float,
        default=5.35,
        help="Enable near-target mode when val loss is below this threshold",
    )
    parser.add_argument(
        "--near_target_lr_ratio",
        type=float,
        default=0.60,
        help="Minimum LR ratio (relative to base LR) while in near-target mode",
    )
    parser.add_argument(
        "--near_target_eval_iter",
        type=int,
        default=20,
        help="Evaluation batches used while in near-target mode",
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.0,
        help="Dropout rate used by the GPT model",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--no_enforce_target",
        action="store_true",
        help="Do not raise error when target_val_loss is not met at target_tokens",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=52000, help="Vocabulary size (should match tokenizer)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Uses a very small model for debugging purposes",
    )

    args = parser.parse_args()

    # Set model configuration
    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": args.vocab_size,
            "context_length": 10,
            "emb_dim": 12,
            "n_heads": 2,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    else:
        GPT_CONFIG_124M = {
            "vocab_size": args.vocab_size,  # Should match tokenizer vocab size
            "context_length": 1024,  # Context length
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": args.drop_rate,  # Dropout rate
            "qkv_bias": False,  # Query-key-value bias
        }

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    # Verify vocab size matches
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != args.vocab_size:
        print(f"Warning: Tokenizer vocab size ({actual_vocab_size}) doesn't match --vocab_size ({args.vocab_size})")
        print(f"Updating model config to use vocab size: {actual_vocab_size}")
        GPT_CONFIG_124M["vocab_size"] = actual_vocab_size

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {output_dir.absolute()}")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        optimizer=optimizer,
        device=device,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        tokenizer=tokenizer,
        data_path=args.data_file,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        target_tokens=args.target_tokens,
        target_val_loss=args.target_val_loss,
        stop_at_target_tokens=not args.no_stop_at_target_tokens,
        enforce_target=not args.no_enforce_target,
        data_fraction=args.data_fraction,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        max_grad_norm=args.max_grad_norm,
        eval_iter=args.eval_iter,
        grad_accum_steps=args.grad_accum_steps,
        near_target_trigger=args.near_target_trigger,
        near_target_lr_ratio=args.near_target_lr_ratio,
        near_target_eval_iter=args.near_target_eval_iter,
    )

    ### START YOUR CODE ###
    # Plot losses if available
    if train_losses:
        epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
        plot_path = output_dir / "loss_curve.pdf"
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        generated_plot = Path("loss.pdf")
        if generated_plot.exists():
            shutil.move(str(generated_plot), str(plot_path))
        print(f"Saved loss curve: {plot_path}")
    else:
        print("No evaluation records found; skipping loss curve plot.")
    
    # Save final model for submission
    final_model_path = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")
    
    ### END YOUR CODE ###
    
    # Print GPU memory usage if CUDA is available
    if torch.cuda.is_available():
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("Training completed!")
