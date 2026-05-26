import argparse
import json
import os

import tiktoken
import torch
import tqdm

from utils import (
    GPTModel,
    generate,
    generate_with_kv_cache,
    text_to_token_ids,
    token_ids_to_text,
)

# Model config
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

CHOOSE_MODEL = "355M"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


def load_model(model_path):
    model = GPTModel(BASE_CONFIG)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    model.eval()

    return model


def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    return instruction_text


def main(args):
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    data = load_data(args.input)

    tokenizer = tiktoken.get_encoding("gpt2")

    # Generate responses
    model = load_model(args.model)  # gpt2-medium355M-sft.pth
    model.to(device)
    model_name = os.path.basename(args.model)  # gpt2-medium355M-sft

    outputs = []
    for entry in tqdm.tqdm(data):
        input_text = format_input(entry)

        token_ids = generate_with_kv_cache(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )
        outputs.append(
            {
                "dataset": entry["dataset"],
                "instruction": entry["instruction"],
                "output": response_text,
                "generator": model_name,
            }
        )

        # print(outputs[-1])
        # if(len(outputs) >= 5):
        #     break

    with open(args.output, "w") as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )

    args = parser.parse_args()
    main(args)
