from __future__ import annotations

import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - handled at runtime
    load_dataset = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


DATASET_NAME = "thesofakillers/jigsaw-toxic-comment-classification-challenge"
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = PROJECT_ROOT / ".model_cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR / "huggingface"))


def main() -> None:
    if load_dataset is None:
        raise RuntimeError(
            "datasets is not installed. Run `pip install -r requirements.txt` first."
        ) from IMPORT_ERROR

    dataset = load_dataset(DATASET_NAME)
    print(f"Dataset: {DATASET_NAME}")
    print("Available splits:")
    for split_name, split_data in dataset.items():
        print(f"  - {split_name}: {len(split_data)} rows")

    first_split = next(iter(dataset))
    print(f"\nColumns in '{first_split}':")
    for column in dataset[first_split].column_names:
        print(f"  - {column}")


if __name__ == "__main__":
    main()
