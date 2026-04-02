from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    from detoxify import Detoxify
except ImportError as exc:  # pragma: no cover - handled at runtime
    Detoxify = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

MODEL_KEY_MAP = {
    "toxic": "toxicity",
    "severe_toxic": "severe_toxicity",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_hate": "identity_attack",
}


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = PROJECT_ROOT / ".model_cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(LOCAL_CACHE_DIR / "torch"))
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR / "huggingface"))


TOXIC_REPLACEMENTS = {
    r"\bidiot\b": "***",
    r"\bdumb\b": "***",
    r"\bstupid\b": "***",
    r"\bmoron\b": "***",
    r"\bshut up\b": "please stay calm",
    r"\bfool\b": "***",
    r"\bhate you\b": "disagree with you",
    r"\bhell\b": "***",
    r"\bjerk\b": "***",
    r"\bbastard\b": "***",
    r"\btrash\b": "***",
    r"\bgarbage\b": "***",
    r"\bkill yourself\b": "please seek support and take a step back",
    r"\bi will kill you\b": "i am very angry with you",
    r"\b(?:f+u+c*k+|fk)\b": "***",
    r"\b(?:s+h+i+t+)\b": "***",
    r"\b(?:b+i+t+c+h+)\b": "***",
    r"\b(?:a+s+s+h*o*l+e?)\b": "***",
    r"\bnigga\b": "***",
    r"\bnigger\b": "***",
    r"\bfag\b": "***",
    r"\bfaggot\b": "***",
    r"\bterrorist\b": "***",
}


@dataclass
class AnalysisResult:
    original_text: str
    cleaned_text: str
    toxic_rating: float
    scores: Dict[str, float]
    predicted_categories: List[str]
    is_toxic: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "toxic_rating": round(self.toxic_rating, 2),
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "predicted_categories": self.predicted_categories,
            "is_toxic": self.is_toxic,
        }


class ToxicCommentSystem:
    def __init__(self, threshold: float = 0.5, model_name: str = "original") -> None:
        if Detoxify is None:
            raise RuntimeError(
                "detoxify is not installed. Run `pip install -r requirements.txt` first."
            ) from IMPORT_ERROR

        self.threshold = threshold
        self.model = Detoxify(model_name)

    def analyze(self, text: str) -> AnalysisResult:
        stripped = text.strip()
        if not stripped:
            raise ValueError("Please enter a non-empty sentence.")

        raw_scores = self.model.predict(stripped)
        scores = {
            label: float(raw_scores[MODEL_KEY_MAP[label]])
            for label in LABELS
        }
        predicted_categories = [
            label for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score >= self.threshold
        ]

        toxic_rating = max(scores.values()) * 100
        is_toxic = len(predicted_categories) > 0
        cleaned_text = self._rewrite_sentence(stripped, is_toxic)

        return AnalysisResult(
            original_text=stripped,
            cleaned_text=cleaned_text,
            toxic_rating=toxic_rating,
            scores=scores,
            predicted_categories=predicted_categories,
            is_toxic=is_toxic,
        )

    def _rewrite_sentence(self, text: str, is_toxic: bool) -> str:
        if not is_toxic:
            return text

        cleaned = text
        for pattern, replacement in TOXIC_REPLACEMENTS.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = self._soften_tone(cleaned)
        return self._finalize_sentence(cleaned)

    @staticmethod
    def _soften_tone(text: str) -> str:
        softened = text
        softened = re.sub(r"\byou are\b", "you seem", softened, flags=re.IGNORECASE)
        softened = re.sub(r"\bI hate\b", "I do not like", softened, flags=re.IGNORECASE)
        softened = re.sub(r"\bThis is unpleasant\b", "This is not ideal", softened, flags=re.IGNORECASE)
        softened = re.sub(r"\byou seem \*{3}\b", "you seem frustrated", softened, flags=re.IGNORECASE)
        return softened

    @staticmethod
    def _finalize_sentence(text: str) -> str:
        finalized = re.sub(r"\b(a|an) \*{3}\b", "***", text, flags=re.IGNORECASE)
        finalized = re.sub(r"(?:\*{3}\s+){2,}", "*** ", finalized, flags=re.IGNORECASE)
        finalized = re.sub(r"\s+", " ", finalized).strip()
        if not finalized:
            return "Please use respectful language."
        return finalized[0].upper() + finalized[1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a sentence into the 6 Jigsaw toxic comment categories."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="A sentence to analyze. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for selecting a toxic category. Default: 0.5",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )
    return parser


def print_result(result: AnalysisResult) -> None:
    print(f"Input sentence      : {result.original_text}")
    print(f"Toxic rating        : {result.toxic_rating:.2f}/100")
    print(f"Detected categories : {', '.join(result.predicted_categories) if result.predicted_categories else 'normal'}")
    print("Category scores:")
    for label, score in result.scores.items():
        print(f"  - {label:<13} {score:.4f}")
    print(f"Safe sentence       : {result.cleaned_text}")


def interactive_loop(system: ToxicCommentSystem, as_json: bool) -> None:
    print("Type a sentence to analyze. Type 'exit' to stop.")
    while True:
        user_text = input("\nEnter sentence: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return

        try:
            result = system.analyze(user_text)
        except ValueError as exc:
            print(exc)
            continue

        if as_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_result(result)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    system = ToxicCommentSystem(threshold=args.threshold)

    if args.text:
        result = system.analyze(args.text)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_result(result)
        return

    interactive_loop(system, as_json=args.json)


if __name__ == "__main__":
    main()
