#!/usr/bin/env python3
"""Audit Assignment 3 deliverables and flag blockers before submission."""

import argparse
import glob
import json
import re
import socket
from pathlib import Path


def exists(path: Path):
    return "PASS" if path.exists() else "FAIL"


def find_any(pattern: str, root: Path):
    return sorted(glob.glob(str(root / pattern)))


def can_reach_huggingface(host: str = "huggingface.co", port: int = 443, timeout: float = 2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_run_pretrain(path: Path):
    if not path.exists():
        return {"status": "FAIL", "reason": "run_pretrain.py missing"}

    txt = path.read_text(encoding="utf-8")
    checks = {
        "has_global_step": bool(re.search(r"global_step\s*=\s*0", txt)),
        "increments_global_step": "global_step += 1" in txt,
        "tracks_tokens_seen": "tokens_seen +=" in txt,
        "supports_eval_freq": "eval_freq" in txt and "evaluate_model" in txt,
        "supports_ckpt_freq": "save_ckpt_freq" in txt and "torch.save(model.state_dict()" in txt,
        "has_target_tokens": "--target_tokens" in txt,
        "has_target_val_loss": "--target_val_loss" in txt,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {"status": status, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="Audit CS310 A3 submission readiness")
    parser.add_argument("--root", default=".", help="A3 folder path")
    parser.add_argument("--report", default="A3_submission_audit.json", help="Output JSON file")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    required_files = {
        "preprocess_script": root / "preprocess_wikizh.py",
        "tokenizer_train_script": root / "train_tokenizer_from_scratch.py",
        "compare_script": root / "compare_tokenizers.py",
        "pretrain_script": root / "run_pretrain.py",
        "tokenizer_json": root / "wikizh_tokenizer_whitespace.json",
        "corpus": root / "wikizh.txt",
    }

    file_status = {k: exists(v) for k, v in required_files.items()}
    run_pretrain_status = check_run_pretrain(required_files["pretrain_script"])

    model_candidates = find_any("model_checkpoints*/model_final.pth", root)
    curve_candidates = find_any("model_checkpoints*/loss_curve.pdf", root)

    report_md_candidates = [
        root / "A3_report_draft.md",
        root / "A3_report.md",
    ]
    report_pdf_candidates = [
        root / "A3_pretrain.pdf",
        root / "report.pdf",
    ]

    report_artifact = {
        "has_markdown_report": any(p.exists() for p in report_md_candidates),
        "has_pdf_report": any(p.exists() for p in report_pdf_candidates),
    }

    report_placeholders = {
        "name_placeholder": False,
        "id_placeholder": False,
        "token_to_fill": False,
    }

    report_path = None
    for p in report_md_candidates:
        if p.exists():
            report_path = p
            break
    if report_path is not None:
        report_txt = report_path.read_text(encoding="utf-8")
        report_placeholders["name_placeholder"] = "[Your Name]" in report_txt
        report_placeholders["id_placeholder"] = "[Your ID]" in report_txt
        report_placeholders["token_to_fill"] = "[TO_FILL]" in report_txt

    has_full_training_model = any("smoke" not in p.lower() for p in model_candidates)
    has_full_training_curve = any("smoke" not in p.lower() for p in curve_candidates)
    hf_reachable = can_reach_huggingface()

    blockers = []
    if "FAIL" in file_status.values():
        blockers.append("Some required files are missing.")
    if run_pretrain_status["status"] == "FAIL":
        blockers.append("run_pretrain.py missing required training-loop logic.")
    if not model_candidates:
        blockers.append("No model_final.pth found under model_checkpoints*/")
    if not curve_candidates:
        blockers.append("No loss_curve.pdf found under model_checkpoints*/")
    if not has_full_training_model:
        blockers.append("Only smoke model artifact found; full training model artifact missing.")
    if not has_full_training_curve:
        blockers.append("Only smoke loss curve found; full training curve missing.")
    if not (report_artifact["has_markdown_report"] or report_artifact["has_pdf_report"]):
        blockers.append("No report artifact found (markdown or pdf).")
    if report_placeholders["name_placeholder"] or report_placeholders["id_placeholder"] or report_placeholders["token_to_fill"]:
        blockers.append("Report draft still contains placeholders.")
    if not hf_reachable:
        blockers.append("Current node cannot reach huggingface.co; compare_tokenizers.py screenshot may require a network-enabled node.")

    summary = {
        "root": str(root),
        "required_files": file_status,
        "run_pretrain": run_pretrain_status,
        "model_candidates": model_candidates,
        "curve_candidates": curve_candidates,
        "has_full_training_model": has_full_training_model,
        "has_full_training_curve": has_full_training_curve,
        "huggingface_reachable": hf_reachable,
        "report_artifact": report_artifact,
        "report_placeholders": report_placeholders,
        "ready_for_submission": len(blockers) == 0,
        "blockers": blockers,
    }

    out = root / args.report
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
