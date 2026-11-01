from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
from pathlib import Path

from dotenv import load_dotenv

from .common import load_config, read_jsonl, resolve_path


TRAIN_MODULES = ("torch", "transformers", "peft", "accelerate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check readiness before a pilot or training run.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    load_dotenv()
    config = load_config(args.config)
    data = config["data"]
    files = {
        split: resolve_path(data[f"{split}_file"])
        for split in ("train", "dev", "test", "svamp")
    }
    missing_modules = [name for name in TRAIN_MODULES if importlib.util.find_spec(name) is None]
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "data": {
            name: {"path": str(path), "exists": path.exists(), "rows": len(read_jsonl(path))}
            for name, path in files.items()
        },
        "teacher": {
            "base_url_set": bool(os.getenv("TEACHER_BASE_URL")),
            "model_set": bool(os.getenv("TEACHER_MODEL")),
            "api_key_set": bool(os.getenv("TEACHER_API_KEY")),
        },
        "training": {
            "missing_modules": missing_modules,
            "student_model": config["student"]["model"],
        },
    }
    report["ready_for_teacher_pilot"] = (
        all(item["exists"] for item in report["data"].values())
        and all(report["teacher"].values())
    )
    report["ready_for_training"] = not missing_modules and Path(
        resolve_path(data["variants_dir"]) / "answer_only.jsonl"
    ).exists()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
