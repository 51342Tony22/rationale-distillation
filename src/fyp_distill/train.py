from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .common import configured_variants, load_config, read_jsonl, resolve_path


class CompletionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            prefix = row["prompt"].rstrip() + "\n"
            full_text = prefix + row["completion"].strip() + tokenizer.eos_token
            full = tokenizer(full_text, truncation=True, max_length=max_length)
            prompt_tokens = tokenizer(prefix, truncation=True, max_length=max_length)["input_ids"]
            labels = full["input_ids"].copy()
            prompt_length = min(len(prompt_tokens), len(labels))
            labels[:prompt_length] = [-100] * prompt_length
            if all(label == -100 for label in labels):
                continue
            self.examples.append({**full, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


@dataclass
class CompletionCollator:
    tokenizer: object

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune one rationale condition.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    config = load_config(args.config)
    if args.variant not in configured_variants(config):
        raise SystemExit(f"Variant {args.variant!r} is not enabled in {args.config}.")
    seed = args.seed if args.seed is not None else config["seed"]
    seed_everything(seed)
    student = config["student"]
    settings = config["training"]
    model_name = args.model or student["model"]
    data_path = resolve_path(config["data"]["variants_dir"]) / f"{args.variant}.jsonl"
    rows = read_jsonl(data_path)
    if not rows:
        raise SystemExit(f"No training rows found at {data_path}. Run build_variants first.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=student.get("trust_remote_code", False)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = CompletionDataset(rows, tokenizer, max_length=student["max_length"])
    if not dataset:
        raise SystemExit("All examples were truncated before the completion; increase max_length.")

    model_kwargs = {
        "trust_remote_code": student.get("trust_remote_code", False),
        "dtype": "auto",
    }
    if torch.backends.mps.is_available():
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=settings["lora_r"],
            lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"],
            target_modules="all-linear",
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    output_dir = resolve_path("outputs") / args.variant / f"seed_{seed}"
    use_bf16 = torch.backends.mps.is_available() or (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        num_train_epochs=settings["num_train_epochs"],
        per_device_train_batch_size=settings["per_device_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        warmup_ratio=settings["warmup_ratio"],
        weight_decay=settings["weight_decay"],
        logging_steps=settings["logging_steps"],
        save_steps=settings["save_steps"],
        save_total_limit=2,
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        torch_empty_cache_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=CompletionCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metadata = {
        "variant": args.variant,
        "seed": seed,
        "base_model": model_name,
        "source_rows": len(rows),
        "tokenized_rows": len(dataset),
        "unique_questions": len({row["source_id"] for row in rows}),
        "max_steps_override": args.max_steps,
        "num_train_epochs": settings["num_train_epochs"],
        "effective_batch_size": (
            settings["per_device_batch_size"] * settings["gradient_accumulation_steps"]
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
