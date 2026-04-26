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
    train_data = None
    test_data = None
    val_data = None

    train_loader = None
    test_loader = None
    val_loader = None
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
            instruction_plus_input = None

            # Format the response
            response_text = f"\n\n### Response:\n{entry['output']}"

            # Concatenate the above two strings
            full_text = None

            # Tokenize the full text, and append to self.encoded_texts
            pass
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
            instruction_plus_input = None

            # Format the response
            response_text = f"\n\n### Response:\n{entry['output']}"

            # Concatenate the above two strings
            full_text = None

            # Tokenize the full text, and append to self.encoded_texts
            pass

            # New: collect instruction lengths, and append to self.instruction_lengths
            instruction_length = None
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
        padded = None

        # Truncate the last token for inputs
        # Shift +1 to the right for targets
        inputs = None
        targets = None

        # Replace all but the first padding tokens in targets with ignore_index
        mask_indices = None
        targets[mask_indices] = None
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
    inputs_tensor = None
    targets_tensor = None
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
    pass
    ### END YOUR CODE ###


def train_model(model, optimizer, device, n_epochs, batch_size, train_loader, val_loader):
    ### START YOUR CODE ###
    train_losses = []
    val_losses = []
    pass
    ### END YOUR CODE ###
    return train_losses, val_losses


def generate(model, input_ids, max_new_tokens:int=256):
    ### START YOUR CODE ###
    idx = []
    pass
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
    model = None
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
    pass    
    ### END YOUR CODE ###

    # Save the model
    ### START YOUR CODE ###
    pass
    ### END YOUR CODE ###

    # Plot the training and validation losses
    ### START YOUR CODE ###
    pass
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