import argparse
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
import tiktoken
from utils import GPTModel


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."

        ### START YOUR CODE ###
        # Format the instruction
        f"\n\n### Instruction:\n{entry['instruction']}"
        ### END YOUR CODE ###
    )

    ### START YOUR CODE ###
    # Format the input
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    ### END YOUR CODE ###

    return instruction_text + input_text


def init_data_loaders(data, tokenizer, batch_size, dataset_class, collate_fn):
    # Split data into train_data, test_data, val_data
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    ### START YOUR CODE ###
    # Split the data into three parts
    train_data = data[:train_portion]
    test_data = data[train_portion: train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # If caller accidentally passed arguments in a different order, try to be tolerant
    # Expectation: batch_size is an int, dataset_class is a Dataset class, collate_fn is callable
    # If batch_size is not int, assume default batch size
    if not isinstance(batch_size, int):
        batch_size = 8

    # Construct datasets
    train_dataset = dataset_class(train_data, tokenizer)
    test_dataset = dataset_class(test_data, tokenizer)
    val_dataset = dataset_class(val_data, tokenizer)

    # Construct loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    ### END YOUR CODE ###

    return train_loader, test_loader, val_loader


# InstructionDataset class, without masking the instruction_plus_input positions.
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            ### START YOUR CODE ###
            # Format the instruction and input, by calling `format_input()`
            instruction_plus_input = format_input(entry)

            # Format the response
            response_text = f"\n\n### Response:\n{entry['output']}"

            # Concatenate the above two strings
            full_text = instruction_plus_input + response_text

            # Tokenize the full text, and append to self.encoded_texts
            token_ids = tokenizer.encode(full_text)
            self.encoded_texts.append(torch.tensor(token_ids, dtype=torch.long))
            ### END YOUR CODE ###

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# InstructionDatasetMask class, with masking the instruction_plus_input positions.
class InstructionDatasetMask(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # New: Separate list for instruction lengths
        self.instruction_lengths = []
        self.encoded_texts = []

        for entry in data:
            ### START YOUR CODE ###
            # Format the instruction and input, by calling `format_input()`
            instruction_plus_input = format_input(entry)

            # Format the response
            response_text = f"\n\n### Response:\n{entry['output']}"

            # Concatenate the above two strings
            full_text = instruction_plus_input + response_text

            # Tokenize the full text, and append to self.encoded_texts
            token_ids = tokenizer.encode(full_text)
            self.encoded_texts.append(torch.tensor(token_ids, dtype=torch.long))

            # New: collect instruction lengths, and append to self.instruction_lengths
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            ### END YOUR CODE ###

    def __getitem__(self, index):
        # New: return both instruction lengths and texts separately
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# Custom collate function without masking
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
    ):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_list, targets_list = [], []

    for item in batch:
        ### START YOUR CODE ###
        # Pad sequence to batch_max_length
        seq = item
        padded = torch.full((batch_max_length,), pad_token_id, dtype=torch.long)
        padded[: len(seq)] = seq

        # Truncate the last token for inputs, targets are shifted by +1
        inputs = padded[:-1].clone()
        targets = padded[1:].clone()

        # Replace padding token ids in targets with ignore_index
        mask_indices = targets == pad_token_id
        targets[mask_indices] = ignore_index
        ### END YOUR CODE ###

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_list.append(inputs)
        targets_list.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    ### START YOUR CODE ###
    # Hint: call torch.stack()
    inputs_tensor = torch.stack([x for x in inputs_list]).to(device)
    targets_tensor = torch.stack([x for x in targets_list]).to(device)
    ### END YOUR CODE ###

    return inputs_tensor, targets_tensor


def custom_collcate_fn_mask(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
    ):
    ### START YOUR CODE ###
    # batch elements: (instruction_length, tensor)
    # Determine max length across sequences
    seqs = [item[1] for item in batch]
    instr_lens = [item[0] for item in batch]
    batch_max_length = max(len(s) + 1 for s in seqs)

    inputs_list, targets_list = [], []
    for instr_len, seq in zip(instr_lens, seqs):
        padded = torch.full((batch_max_length,), pad_token_id, dtype=torch.long)
        padded[: len(seq)] = seq

        inputs = padded[:-1].clone()
        targets = padded[1:].clone()

        # Mask padding tokens
        targets[targets == pad_token_id] = ignore_index

        # Mask instruction+input positions in targets (do not learn to predict them)
        if instr_len > 0:
            # targets correspond to tokens shifted by 1, so mask positions [0:instr_len]
            targets[:instr_len] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tensor
    ### END YOUR CODE ###


def train_model(model, optimizer, device, n_epochs, batch_size, train_loader, val_loader):
    ### START YOUR CODE ###
    train_losses = []
    val_losses = []

    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            # If model is a transformers AutoModelForCausalLM-like
            try:
                outputs = model(input_ids=inputs, labels=targets)
                loss = outputs.loss
            except Exception:
                # Fallback: assume model returns logits and needs manual loss
                logits = model(inputs)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(1, n_batches)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_running = 0.0
        val_batches = 0
        if val_loader is not None:
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs = v_inputs.to(device)
                    v_targets = v_targets.to(device)
                    try:
                        v_out = model(input_ids=v_inputs, labels=v_targets)
                        v_loss = v_out.loss
                    except Exception:
                        logits = model(v_inputs)
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        v_loss = loss_fct(logits.view(-1, logits.size(-1)), v_targets.view(-1))

                    val_running += v_loss.item()
                    val_batches += 1
            avg_val_loss = val_running / max(1, val_batches)
        else:
            avg_val_loss = None

        val_losses.append(avg_val_loss)

    return train_losses, val_losses
    ### END YOUR CODE ###
    return train_losses, val_losses


def generate(model, input_ids, max_new_tokens:int=256):
    ### START YOUR CODE ###
    idx = []
    pass
    # Accept either list of ids or tensor
    device = next(model.parameters()).device
    if isinstance(input_ids, list):
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    elif isinstance(input_ids, torch.Tensor):
        input_tensor = input_ids.unsqueeze(0).to(device)
    else:
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    try:
        # transformers models have .generate
        gen = model.generate(input_tensor, max_new_tokens=max_new_tokens)
        idx = gen[0].tolist()
    except Exception:
        # naive autoregressive sampling fallback
        idx = input_tensor[0].tolist()
        model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids_tensor = torch.tensor([idx], dtype=torch.long).to(device)
                logits = model(input_ids_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[0, -1, :]
                next_id = int(torch.argmax(next_token_logits).item())
                idx.append(next_id)
    ### END YOUR CODE ###
    return idx


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    allowed_max_length = 1024
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.mask_instructions == 1:
        CustomDataset = InstructionDatasetMask
        customized_collate_fn = partial(custom_collcate_fn_mask, allowed_max_length=allowed_max_length, device=device)
    elif args.mask_instructions == 0:
        CustomDataset = InstructionDataset
        customized_collate_fn = partial(custom_collate_fn, allowed_max_length=allowed_max_length, device=device)
    
    # Load the data
    with open(args.data, "r", encoding="utf-8") as file:
        data = json.load(file)
    train_loader, test_loader, val_loader = init_data_loaders(data, tokenizer, CustomDataset, customized_collate_fn)
    print("Data loaded.")
    
    # Configure the model
    BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
    }
    model_configs = {
        "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}
    }
    BASE_CONFIG.update(model_configs[args.model_config])

    # Load the pretrained model
    ### START YOUR CODE ###
    # Prefer the provided weights if available, else initialize model from utils.GPTModel
    model = GPTModel(BASE_CONFIG)
    if args.model_path:
        try:
            state = torch.load(args.model_path, map_location=device)
            # allow either full model saved or state_dict
            if isinstance(state, dict) and any(k.startswith('transformer') or k.startswith('emb') for k in state.keys()):
                model.load_state_dict(state)
            else:
                try:
                    model.load_state_dict(state)
                except Exception:
                    # fallback: assume the file is a full model
                    model = state
        except Exception:
            pass
    model.to(device)
    ### END YOUR CODE ###
    print("Pretrained model loaded.")

    # Training hyperparameters
    num_epochs = 2
    batch_size = 8
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    # Run SFT 
    ### START YOUR CODE ###
    # Main training loop
    num_epochs = args.num_epochs if hasattr(args, 'num_epochs') else num_epochs
    train_losses, val_losses = train_model(model, optimizer, device, num_epochs, batch_size, train_loader, val_loader)
    ### END YOUR CODE ###
    ### END YOUR CODE ###

    # Save the model
    ### START YOUR CODE ###
    try:
        torch.save(model.state_dict(), args.save_path)
    except Exception:
        torch.save(model, args.save_path)
    ### END YOUR CODE ###

    # Plot the training and validation losses
    ### START YOUR CODE ###
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_losses, label='train')
        if any(v is not None for v in val_losses):
            plt.plot(val_losses, label='val')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('sft_losses.png')
        plt.close()
    except Exception:
        # If plotting fails, just print the losses
        print('Train losses:', train_losses)
        print('Val losses:', val_losses)
    ### END YOUR CODE ###


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="alpaca_data.json")
    parser.add_argument("--model_config", type=str, default="124M")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="sft_model.pth")
    parser.add_argument("--mask_instructions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--generate_responses", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    main(args)