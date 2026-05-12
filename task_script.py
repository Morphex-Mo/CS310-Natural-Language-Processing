import json
import torch
from transformers import AutoTokenizer, AutoConfig

def get_info(model_path, prompt_text):
    print(f'--- Model: {model_path} ---')
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Use config vocab size
    config_vocab_size = config.vocab_size
    tokenizer_len = len(tokenizer)
    
    # Tokenize prompt
    tokens = tokenizer(prompt_text)['input_ids']
    max_token_id = max(tokens) if tokens else 0
    
    # Embedding shape (from config if possible)
    # Most models have (vocab_size, hidden_size)
    hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 'N/A'))
    
    print(f'Tokenizer length: {tokenizer_len}')
    print(f'Model config vocab_size: {config_vocab_size}')
    print(f'Max token ID in prompt: {max_token_id}')
    print(f'Embedding matrix shape (est.): ({config_vocab_size}, {hidden_size})')
    
    if max_token_id >= config_vocab_size:
        print('CRITICAL: Max token ID exceeds embedding range!')
    else:
        print('Status: Max token ID is within embedding range.')
    print()

# Typical MMLU prompt example construction pattern from the notebook
# Using a generic subject prompt based on the structure seen in lab11_prompt.ipynb
# Since we don't have the JSON files, we construct a representative string.
test_prompt = 'The following are multiple choice questions (with answers) about high school mathematics.\n\nWhat is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer: B'

models = ['lab/lab8/gpt2-mini', 'lab/lab8/Qwen3-0.6B-Base']
for m in models:
    try:
        get_info(m, test_prompt)
    except Exception as e:
        print(f'Error processing {m}: {e}')
