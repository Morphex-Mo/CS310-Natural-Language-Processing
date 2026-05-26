import argparse
import json
import os
import time
from functools import partial
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Dataset

from utils import GPTModel

# =========================
# Device utilities
# =========================
def get_device() -> torch.device:
    """Select the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            return torch.device("mps")
    return torch.device("cpu")

# =========================
# Data formatting utilities
# =========================
def format_input(entry: Dict[str, str]) -> str:
    """Format the instruction and optional input into a prompt."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def format_response(response: str) -> str:
    """Format the response (chosen or rejected) into the expected structure."""
    return f"\n\n### Response:\n{response}"

# =========================
# Dataset class
# =========================
class PreferenceDataset(Dataset):
    """Dataset that pre‑tokenizes prompts, chosen and rejected responses."""

    def __init__(self, data: List[Dict[str, str]], tokenizer):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            # Build full texts
            prompt = format_input(entry)
            chosen = format_response(entry['chosen'])
            rejected = format_response(entry['rejected'])

            chosen_full_text = prompt + '\n\n' + chosen
            rejected_full_text = prompt + '\n\n' + rejected

            # Tokenize
            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)

# =========================
# Collate function
# =========================
def custom_collate_fn(
    batch: List[Dict[str, List[int]]],
    pad_token_id: int = 50256,
    allowed_max_length: Optional[int] = None,
    mask_prompt_tokens: bool = True,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Pad sequences, create attention masks, and optionally mask prompt tokens."""
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": [],
    }

    # Determine the maximum length among all chosen and rejected sequences (+1 for safety)
    max_length_common = 0
    if batch:
        for key in ("chosen", "rejected"):
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ("chosen", "rejected"):
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded), dtype=torch.bool)
            mask[len(sequence):] = False

            if mask_prompt_tokens:
                # Also mask the prompt tokens and the two newlines that follow
                mask[:prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    for key in ("chosen", "rejected", "chosen_mask", "rejected_mask"):
        tensor_stack = torch.stack(batch_data[key])
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]
        batch_data[key] = tensor_stack.to(device)

    return batch_data

# =========================
# Data loaders creation
# =========================
def init_data_loaders(
    data: List[Dict[str, str]],
    tokenizer,
    batch_size: int,
    collate_fn
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split data into train/test/val and create DataLoaders."""
    train_portion = int(len(data) * 0.85)   # 85% training
    test_portion = int(len(data) * 0.1)    # 10% testing
    # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    train_dataset = PreferenceDataset(train_data, tokenizer)
    test_dataset = PreferenceDataset(test_data, tokenizer)
    val_dataset = PreferenceDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    return train_loader, test_loader, val_loader

# =========================
# DPO Loss components
# =========================
def compute_dpo_loss(
    model_chosen_logprobs: torch.Tensor,
    model_rejected_logprobs: torch.Tensor,
    reference_chosen_logprobs: torch.Tensor,
    reference_rejected_logprobs: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss and reward statistics."""
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def compute_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    selection_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per‑sequence average log‑probabilities, optionally masked."""
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(
        input=log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()
        selected_log_probs = selected_log_probs * mask
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)
        return avg_log_prob
    else:
        return selected_log_probs.mean(-1)

def compute_dpo_loss_batch(
    batch: Dict[str, torch.Tensor],
    policy_model,
    reference_model,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss for a single batch."""
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"],
    )
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"],
    )

    with torch.no_grad():
        ref_chosen_log_probas = compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"],
        )
        ref_rejected_log_probas = compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"],
        )

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta,
    )
    return loss, chosen_rewards, rejected_rewards

# =========================
# Model building helpers
# =========================
def build_base_config() -> Dict[str, Any]:
    """Return base configuration for GPT‑2 355M."""
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model_configs = {
        "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    }
    BASE_CONFIG.update(model_configs["355M"])
    return BASE_CONFIG

def build_models_from_sft(
    sft_path: str,
    base_config: Dict[str, Any],
    device: torch.device
) -> Tuple[GPTModel, GPTModel]:
    """Load SFT checkpoint into policy and reference models."""
    state = torch.load(sft_path, map_location=device)
    policy = GPTModel(base_config).to(device)
    reference = GPTModel(base_config).to(device)
    policy.load_state_dict(state)
    reference.load_state_dict(state)
    reference.eval()
    return policy, reference

# =========================
# Training loop
# =========================
def train_model_dpo_simple(
    policy_model,
    reference_model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    num_epochs: int,
    beta: float,
    eval_freq: int,
    eval_iter: int,
    tokenizer,
) -> Dict[str, List[float]]:
    """Main DPO training loop."""
    history = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_processed": [],
    }
    tokens_processed, global_step = 0, -1

    for epoch in range(num_epochs):
        policy_model.train()
        reference_model.eval()

        for batch in train_loader:
            optimizer.zero_grad()

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta,
            )

            loss.backward()
            optimizer.step()

            tokens_processed += batch["chosen"].numel() + batch["rejected"].numel()
            global_step += 1

            if global_step % eval_freq == 0:
                policy_model.eval()

                # inner evaluation function
                def _evaluate(loader):
                    total_loss = 0.0
                    total_chosen_rewards = 0.0
                    total_rejected_rewards = 0.0
                    num_batches = 0

                    with torch.no_grad():
                        for eval_step, eval_batch in enumerate(loader):
                            if eval_step >= eval_iter:
                                break

                            eval_loss, eval_chosen_rewards, eval_rejected_rewards = (
                                compute_dpo_loss_batch(
                                    batch=eval_batch,
                                    policy_model=policy_model,
                                    reference_model=reference_model,
                                    beta=beta,
                                )
                            )

                            total_loss += eval_loss.item()
                            total_chosen_rewards += eval_chosen_rewards.item()
                            total_rejected_rewards += eval_rejected_rewards.item()
                            num_batches += 1

                    if num_batches == 0:
                        return 0.0, 0.0, 0.0

                    return (
                        total_loss / num_batches,
                        total_chosen_rewards / num_batches,
                        total_rejected_rewards / num_batches,
                    )

                train_loss, train_chosen_reward, train_rejected_reward = _evaluate(train_loader)
                val_loss, val_chosen_reward, val_rejected_reward = _evaluate(val_loader)

                history["train_losses"].append(train_loss)
                history["train_chosen_rewards"].append(train_chosen_reward)
                history["train_rejected_rewards"].append(train_rejected_reward)
                history["val_losses"].append(val_loss)
                history["val_chosen_rewards"].append(val_chosen_reward)
                history["val_rejected_rewards"].append(val_rejected_reward)
                history["tokens_processed"].append(tokens_processed)

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Train chosen reward {train_chosen_reward:.3f}, "
                    f"Train rejected reward {train_rejected_reward:.3f}, "
                    f"Val loss {val_loss:.3f}, "
                    f"Val chosen reward {val_chosen_reward:.3f}, "
                    f"Val rejected reward {val_rejected_reward:.3f}"
                )

                policy_model.train()

    return history

# =========================
# Plotting utility
# =========================
def save_training_curves(history: Dict[str, List[float]], train_loader: DataLoader) -> None:
    """Plot loss and reward margin curves using matplotlib."""
    if not history["tokens_processed"]:
        return

    total_tokens_per_epoch = sum(
        batch["chosen"].numel() + batch["rejected"].numel() for batch in train_loader
    )
    epochs_seen = [tokens / total_tokens_per_epoch for tokens in history["tokens_processed"]]

    reward_margins = [
        train_chosen - train_rejected
        for train_chosen, train_rejected in zip(
            history["train_chosen_rewards"], history["train_rejected_rewards"]
        )
    ]
    val_reward_margins = [
        val_chosen - val_rejected
        for val_chosen, val_rejected in zip(
            history["val_chosen_rewards"], history["val_rejected_rewards"]
        )
    ]

    # Loss plot
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs_seen, history["train_losses"], label="Training loss")
    ax1.plot(epochs_seen, history["val_losses"], linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(history["tokens_processed"], history["train_losses"], alpha=0)
    ax2.set_xlabel("Tokens processed")

    fig.tight_layout()
    fig.savefig("dpo_training_loss.png", dpi=200)
    plt.close(fig)

    # Reward margin plot
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs_seen, reward_margins, label="Training reward margin")
    ax1.plot(epochs_seen, val_reward_margins, linestyle="-.", label="Validation reward margin")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Reward margin")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(history["tokens_processed"], reward_margins, alpha=0)
    ax2.set_xlabel("Tokens processed")

    fig.tight_layout()
    fig.savefig("dpo_reward_margin.png", dpi=200)
    plt.close(fig)

    print("Saved training curves to dpo_training_loss.png and dpo_reward_margin.png")

# =========================
# Main execution
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    # Load dataset
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "instruction-data-with-preference.json")
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print("Number of entries:", len(data))

    # Tokenizer (GPT‑2)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Data loaders
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        mask_prompt_tokens=True,
        allowed_max_length=1024,
    )
    batch_size = 8
    torch.manual_seed(123)

    train_loader, test_loader, val_loader = init_data_loaders(
        data, tokenizer, batch_size, customized_collate_fn
    )

    # Build models
    base_config = build_base_config()
    sft_model_path = os.path.join(script_dir, "sft_model.pth")
    policy_model, reference_model = build_models_from_sft(sft_model_path, base_config, device)
    print("Pretrained model loaded.")

    # Training
    start_time = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)
    num_epochs = 1

    history = train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=reference_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        beta=0.1,
        eval_freq=args.log_interval,
        eval_iter=10,
        tokenizer=tokenizer,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Save policy model
    torch.save(policy_model.state_dict(), "gpt2-medium355M-dpo.pth")
    print("Saved policy model to gpt2-medium355M-dpo.pth")

    # Plot curves
    save_training_curves(history, train_loader)