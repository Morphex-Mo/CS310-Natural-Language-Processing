#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.expanduser('~/.cache/tiktoken')
import json
import gc
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

torch.backends.cudnn.benchmark = True

from utils import GPTModel

# ---------- 硬件配置 ----------
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    _device = torch.device('cuda:0')
else:
    _device = torch.device('cpu')
print('Using device:', _device)
if _device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
    print(f'Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# ---------- 辅助函数 ----------
def build_prompt(item):
    inst = (f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{item['instruction']}")
    inp = f"\n\n### Input:\n{item.get('input')}" if item.get('input') else ""
    return inst + inp

def extract_response(text):
    text = text.strip()
    if '### Response:' in text:
        text = text.split('### Response:', 1)[-1].strip()
    if '### Instruction:' in text:
        text = text.split('### Instruction:', 1)[0].strip()
    return text

def produce_tokens(model, input_ids, max_new=100, stop_id=None):
    model.eval()
    seq = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(seq)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
            if stop_id is not None and next_tok.item() == stop_id:
                break
    return seq

# ---------- 数据集类（重命名）----------
class _DatasetV1(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self._enc = []
        for entry in data:
            prompt = build_prompt(entry)
            response = f"\n\n### Response:\n{entry.get('output','')}"
            full = prompt + response + '<|endoftext|>'
            self._enc.append(tokenizer.encode(full, allowed_special={'<|endoftext|>'}))

    def __getitem__(self, idx):
        return self._enc[idx]

    def __len__(self):
        return len(self.data)

class _DatasetV2(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self._inst_lens = []
        self._enc = []
        for entry in data:
            prompt = build_prompt(entry)
            response = f"\n\n### Response:\n{entry.get('output','')}"
            full = prompt + response + '<|endoftext|>'
            enc_full = tokenizer.encode(full, allowed_special={'<|endoftext|>'})
            self._enc.append(enc_full)
            enc_inst = tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
            self._inst_lens.append(len(enc_inst))

    def __getitem__(self, idx):
        return self._inst_lens[idx], self._enc[idx]

    def __len__(self):
        return len(self.data)

# ---------- 自定义批处理函数 ----------
def _collate_v1(batch, pad_id=50256, ignore_idx=-100, max_len=None, device='cpu'):
    max_batch_len = max(len(item) + 1 for item in batch)
    inp_list, tgt_list = [], []
    for item in batch:
        padded = item + [pad_id] * (max_batch_len - len(item))
        inp = torch.tensor(padded[:-1], dtype=torch.long)
        tgt = torch.tensor(padded[1:], dtype=torch.long)
        mask = (tgt == pad_id)
        first_pad = mask.nonzero(as_tuple=True)[0]
        if first_pad.numel() > 0:
            mask[first_pad[0]] = False
        tgt[mask] = ignore_idx
        if max_len:
            inp = inp[:max_len]
            tgt = tgt[:max_len]
        inp_list.append(inp)
        tgt_list.append(tgt)
    return torch.stack(inp_list).to(device), torch.stack(tgt_list).to(device)

def _collate_v2(batch, pad_id=50256, ignore_idx=-100, max_len=None, device='cpu'):
    max_batch_len = max(len(item[1]) + 1 for item in batch)
    inp_list, tgt_list = [], []
    for inst_len, item in batch:
        padded = item + [pad_id] * (max_batch_len - len(item))
        inp = torch.tensor(padded[:-1], dtype=torch.long)
        tgt = torch.tensor(padded[1:], dtype=torch.long)
        mask = (tgt == pad_id)
        first_pad = mask.nonzero(as_tuple=True)[0]
        if first_pad.numel() > 0:
            mask[first_pad[0]] = False
        tgt[mask] = ignore_idx
        if inst_len > 1:
            tgt[:inst_len - 1] = ignore_idx
        if max_len:
            inp = inp[:max_len]
            tgt = tgt[:max_len]
        inp_list.append(inp)
        tgt_list.append(tgt)
    return torch.stack(inp_list).to(device), torch.stack(tgt_list).to(device)

# ---------- 数据加载器构造 ----------
def _create_loaders(data, tokenizer, batch_size, ds_class, collate_fn):
    n_train = int(len(data) * 0.85)
    n_test = int(len(data) * 0.1)
    n_val = len(data) - n_train - n_test
    train_ds = ds_class(data[:n_train], tokenizer)
    test_ds = ds_class(data[n_train:n_train + n_test], tokenizer)
    val_ds = ds_class(data[n_train + n_test:], tokenizer)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader, val_loader

# ---------- 训练循环 ----------
def _train_loop(model, opt, device, epochs, train_loader, val_loader,
                use_amp=False, grad_acc=1, max_norm=None, log_every=50,
                patience=2, min_delta=1e-4):
    train_losses, val_losses = [], []
    amp_on = use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        steps_done = 0
        opt.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=amp_on):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            raw_loss = loss.item()
            loss = loss / max(1, grad_acc)
            scaler.scale(loss).backward()

            if step % grad_acc == 0 or step == len(train_loader):
                if max_norm is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running_loss += raw_loss
            steps_done += 1
            if step % log_every == 0 or step == len(train_loader):
                avg_sofar = running_loss / steps_done
                print(f'[Train][Epoch {epoch+1}/{epochs}] Step {step}/{len(train_loader)} | batch_loss={raw_loss:.4f} | avg_train_loss_so_far={avg_sofar:.4f}')

        avg_train = running_loss / steps_done
        train_losses.append(avg_train)

        # 验证
        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                with torch.cuda.amp.autocast(enabled=amp_on):
                    vlogits = model(vx)
                    vloss = criterion(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                val_total += vloss.item()
                val_count += 1
        avg_val = val_total / val_count if val_count else None
        val_losses.append(avg_val)
        print(f'[Epoch {epoch+1}/{epochs}] train_loss={avg_train:.4f} | val_loss={avg_val:.4f}')

        # 早停 + 保存最佳
        if avg_val is not None and avg_val < best_val - min_delta:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f'  -> New best val_loss={best_val:.4f}')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    return train_losses, val_losses, best_state

# ---------- 主实验函数 ----------
def run_experiment(args):
    tokenizer = tiktoken.get_encoding('gpt2')
    pad_id = tokenizer.eot_token

    # 选择数据集和 collate
    if args.mask_instructions == 1:
        DS = _DatasetV2
        collate = partial(_collate_v2, max_len=args.allowed_max_length, device=_device)
    else:
        DS = _DatasetV1
        collate = partial(_collate_v1, max_len=args.allowed_max_length, device=_device)

    with open(args.data, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    train_loader, test_loader, val_loader = _create_loaders(raw_data, tokenizer, args.batch_size, DS, collate)
    print(f"Data loaded. total={len(raw_data)}, train_batches={len(train_loader)}, val_batches={len(val_loader)}, test_batches={len(test_loader)}")

    # 模型配置
    BASE = {'vocab_size': 50257, 'context_length': 1024, 'drop_rate': 0.0, 'qkv_bias': True}
    model_sizes = {
        '124M': {'emb_dim': 768, 'n_layers': 12, 'n_heads': 12},
        '355M': {'emb_dim': 1024, 'n_layers': 24, 'n_heads': 16}
    }
    BASE.update(model_sizes[args.model_config])
    model = GPTModel(BASE)

    if args.model_path:
        print(f'[Load] Loading pretrained weights from {args.model_path}')
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()

    model.to(_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses, val_losses, best_state = [], [], None
    if args.num_epochs > 0:
        train_losses, val_losses, best_state = _train_loop(
            model, optimizer, _device, args.num_epochs, train_loader, val_loader,
            use_amp=args.use_amp, grad_acc=args.grad_accum_steps, max_norm=args.max_grad_norm,
            log_every=args.log_interval, patience=args.early_stop_patience, min_delta=args.min_delta
        )
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), args.save_path)
        print(f'Model saved to {args.save_path}')

        # 绘制损失曲线
        try:
            import matplotlib.pyplot as plt
            epochs_x = list(range(1, len(train_losses) + 1))
            plt.figure(figsize=(6, 4))
            if train_losses:
                plt.plot(epochs_x, train_losses, label='Train Loss')
            if val_losses:
                plt.plot(epochs_x, val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(args.save_path + '.loss.png')
            print('Loss plot saved to', args.save_path + '.loss.png')
            plt.close()
        except Exception as e:
            print('Plotting failed:', e)

    # 生成响应
    responses_file = None
    if args.generate_responses:
        print('Generating responses on test set...')
        n_train = int(len(raw_data) * 0.85)
        n_test = int(len(raw_data) * 0.1)
        test_raw = raw_data[n_train:n_train + n_test]
        total = len(test_raw)
        results = []
        for i, entry in enumerate(test_raw, 1):
            prompt = build_prompt(entry) + '\n\n### Response:\n'
            tok_ids = tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
            inp_tensor = torch.tensor([tok_ids], dtype=torch.long).to(_device)
            gen_ids = produce_tokens(model, inp_tensor, max_new=args.max_new_tokens, stop_id=tokenizer.eot_token)
            gen_text = tokenizer.decode(gen_ids[0].tolist())
            response = gen_text[len(prompt):] if len(gen_text) > len(prompt) else ''
            response = extract_response(response)
            response = response.replace('<|endoftext|>', '').strip()
            if '###' in response:
                response = response.split('###')[0].strip()
            if not response:
                response = '[No response generated]'
            results.append({
                'instruction': entry.get('instruction', ''),
                'input': entry.get('input', ''),
                'output': entry.get('output', ''),
                'model_response': response
            })
            if i % 10 == 0 or i == total:
                print(f'  Generated {i}/{total}')
        responses_file = args.save_path + '.responses.json'
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print('Responses saved to', responses_file)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_path': args.save_path,
        'responses_file': responses_file
    }


if __name__ == '__main__':
    from types import SimpleNamespace

    # 基础参数（与原始完全一致）
    _common_params = {
        'data': 'instruction-data.json',
        'model_config': '355M',
        'model_path': 'gpt2-355M.pth',
        'num_epochs': 3,
        'batch_size': 8,
        'grad_accum_steps': 1,
        'lr': 2e-5,
        'weight_decay': 0.05,
        'use_amp': True,
        'max_grad_norm': 1.0,
        'log_interval': 10,
        'early_stop_patience': 1,
        'min_delta': 1e-4,
        'allowed_max_length': 512,
        'max_new_tokens': 128,
        'generate_responses': 1
    }

    # 实验1：无 mask
    args_no_mask = SimpleNamespace(**{**_common_params, 'mask_instructions': 0, 'save_path': 'sft_no_mask.pth'})
    # 实验2：有 mask
    args_mask = SimpleNamespace(**{**_common_params, 'mask_instructions': 1, 'save_path': 'sft_mask.pth'})

    print('=== Running without instruction masking ===')
    res_no_mask = run_experiment(args_no_mask)
    print('No-mask summary:')
    print('  model saved to', res_no_mask['model_path'])
    print('  responses saved to', res_no_mask['responses_file'])

    print('=== Running WITH instruction masking ===')
    res_mask = run_experiment(args_mask)
    print('Mask summary:')
    print('  model saved to', res_mask['model_path'])
    print('  responses saved to', res_mask['responses_file'])

    # 绘制对比曲线（可选）
    try:
        import matplotlib.pyplot as plt
        x1 = list(range(1, len(res_no_mask['train_losses']) + 1))
        x2 = list(range(1, len(res_mask['train_losses']) + 1))
        plt.figure(figsize=(8, 4))
        if res_no_mask['train_losses']:
            plt.plot(x1, res_no_mask['train_losses'], label='no_mask_train')
        if res_no_mask['val_losses']:
            plt.plot(x1, res_no_mask['val_losses'], label='no_mask_val')
        if res_mask['train_losses']:
            plt.plot(x2, res_mask['train_losses'], label='mask_train')
        if res_mask['val_losses']:
            plt.plot(x2, res_mask['val_losses'], label='mask_val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print('Could not plot comparison:', e)