from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from app import LABELS, ToxicCommentSystem


class ToxicCommentGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Toxic Comment Classifier")
        self.root.geometry("820x700")
        self.root.minsize(760, 620)

        self.system: ToxicCommentSystem | None = None
        self.status_var = tk.StringVar(value="Loading pretrained model...")
        self.rating_var = tk.StringVar(value="Toxic rating: -")
        self.categories_var = tk.StringVar(value="Detected categories: -")
        self.safe_text_var = tk.StringVar(value="-")
        self.threshold_var = tk.DoubleVar(value=0.50)

        self.score_vars = {label: tk.StringVar(value="0.0000") for label in LABELS}

        self._build_layout()
        self._load_model_async()

    def _build_layout(self) -> None:
        self.root.configure(bg="#f4f6fb")

        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)

        title = ttk.Label(
            container,
            text="Toxic Comments Classification",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            container,
            text="Classify a sentence into 6 toxic categories and rewrite harmful text into safer language.",
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        input_frame = ttk.LabelFrame(container, text="Input Sentence", padding=14)
        input_frame.pack(fill="x")

        self.input_text = tk.Text(
            input_frame,
            height=6,
            wrap="word",
            font=("Segoe UI", 11),
            relief="solid",
            borderwidth=1,
        )
        self.input_text.pack(fill="x")

        controls = ttk.Frame(container, padding=(0, 14, 0, 14))
        controls.pack(fill="x")

        ttk.Label(controls, text="Threshold").pack(side="left")

        threshold_spin = ttk.Spinbox(
            controls,
            from_=0.10,
            to=0.95,
            increment=0.05,
            textvariable=self.threshold_var,
            width=8,
            format="%.2f",
        )
        threshold_spin.pack(side="left", padx=(8, 14))

        self.analyze_button = ttk.Button(
            controls,
            text="Analyze Sentence",
            command=self._analyze_async,
            state="disabled",
        )
        self.analyze_button.pack(side="left")

        ttk.Button(controls, text="Clear", command=self._clear).pack(side="left", padx=(10, 0))

        status_label = ttk.Label(
            controls,
            textvariable=self.status_var,
            foreground="#1f4a7c",
        )
        status_label.pack(side="right")

        result_frame = ttk.LabelFrame(container, text="Results", padding=14)
        result_frame.pack(fill="both", expand=True)

        ttk.Label(
            result_frame,
            textvariable=self.rating_var,
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            result_frame,
            textvariable=self.categories_var,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(6, 16))

        score_frame = ttk.Frame(result_frame)
        score_frame.pack(fill="x")

        for index, label in enumerate(LABELS):
            row = ttk.Frame(score_frame)
            row.pack(fill="x", pady=3)

            ttk.Label(row, text=label, width=16).pack(side="left")

            bar = ttk.Progressbar(row, maximum=100, length=360)
            bar.pack(side="left", padx=(0, 10))

            value_label = ttk.Label(row, textvariable=self.score_vars[label], width=10)
            value_label.pack(side="left")

            setattr(self, f"{label}_bar", bar)

        safe_frame = ttk.LabelFrame(container, text="Safe Sentence", padding=14)
        safe_frame.pack(fill="both", expand=True, pady=(14, 0))

        self.safe_text = tk.Text(
            safe_frame,
            height=7,
            wrap="word",
            font=("Segoe UI", 11),
            relief="solid",
            borderwidth=1,
            state="disabled",
        )
        self.safe_text.pack(fill="both", expand=True)

    def _load_model_async(self) -> None:
        thread = threading.Thread(target=self._load_model, daemon=True)
        thread.start()

    def _load_model(self) -> None:
        try:
            system = ToxicCommentSystem(threshold=self.threshold_var.get())
        except Exception as exc:
            self.root.after(0, lambda: self._handle_model_error(exc))
            return

        def finish() -> None:
            self.system = system
            self.analyze_button.config(state="normal")
            self.status_var.set("Model loaded and ready.")

        self.root.after(0, finish)

    def _handle_model_error(self, exc: Exception) -> None:
        self.status_var.set("Failed to load model.")
        messagebox.showerror("Model Error", str(exc))

    def _analyze_async(self) -> None:
        if self.system is None:
            messagebox.showinfo("Please wait", "The model is still loading.")
            return

        sentence = self.input_text.get("1.0", "end").strip()
        if not sentence:
            messagebox.showwarning("Missing input", "Please type a sentence first.")
            return

        try:
            threshold = float(self.threshold_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid threshold", "Please enter a valid threshold between 0 and 1.")
            return

        self.system.threshold = threshold
        self.analyze_button.config(state="disabled")
        self.status_var.set("Analyzing sentence...")

        thread = threading.Thread(target=self._analyze_sentence, args=(sentence,), daemon=True)
        thread.start()

    def _analyze_sentence(self, sentence: str) -> None:
        try:
            result = self.system.analyze(sentence) if self.system is not None else None
        except Exception as exc:
            self.root.after(0, lambda: self._handle_analysis_error(exc))
            return

        if result is not None:
            self.root.after(0, lambda: self._update_results(result))

    def _handle_analysis_error(self, exc: Exception) -> None:
        self.analyze_button.config(state="normal")
        self.status_var.set("Analysis failed.")
        messagebox.showerror("Analysis Error", str(exc))

    def _update_results(self, result) -> None:
        self.rating_var.set(f"Toxic rating: {result.toxic_rating:.2f}/100")
        categories = ", ".join(result.predicted_categories) if result.predicted_categories else "normal"
        self.categories_var.set(f"Detected categories: {categories}")

        for label in LABELS:
            score = result.scores[label]
            self.score_vars[label].set(f"{score:.4f}")
            getattr(self, f"{label}_bar")["value"] = score * 100

        self.safe_text.config(state="normal")
        self.safe_text.delete("1.0", "end")
        self.safe_text.insert("1.0", result.cleaned_text)
        self.safe_text.config(state="disabled")

        self.status_var.set("Analysis complete.")
        self.analyze_button.config(state="normal")

    def _clear(self) -> None:
        self.input_text.delete("1.0", "end")
        self.safe_text.config(state="normal")
        self.safe_text.delete("1.0", "end")
        self.safe_text.insert("1.0", "-")
        self.safe_text.config(state="disabled")
        self.rating_var.set("Toxic rating: -")
        self.categories_var.set("Detected categories: -")
        for label in LABELS:
            self.score_vars[label].set("0.0000")
            getattr(self, f"{label}_bar")["value"] = 0
        self.status_var.set("Ready." if self.system is not None else "Loading pretrained model...")


def main() -> None:
    root = tk.Tk()
    ToxicCommentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
