from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


INPUT_MODES = ("q_only", "y_only", "q_y")


def format_pair_text(row: dict, mode: str) -> str:
    if mode not in INPUT_MODES:
        raise ValueError(f"unknown input mode: {mode}")
    query = str(row.get("user_query") or "")
    answer = str(row.get("target_model_answer") or "")
    if mode == "q_only":
        return f"Question: {query}"
    if mode == "y_only":
        return f"Answer: {answer}"
    return f"Question: {query}\nAnswer: {answer}"


def pair_texts(rows: list[dict], mode: str) -> list[str]:
    return [format_pair_text(row, mode) for row in rows]


@dataclass
class TrainResult:
    threshold: float
    metrics: dict[str, float]
    scores: list[float]


class XLMRPairCrossEncoder:
    def __init__(
        self,
        model_name: str = "FacebookAI/xlm-roberta-base",
        max_length: int = 512,
        batch_size: int = 8,
        lr: float = 2e-5,
        epochs: int = 1,
        seed: int = 0,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, rows: list[dict], labels: list[str], mode: str) -> None:
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import get_linear_schedule_with_warmup

        torch = self.torch
        texts = pair_texts(rows, mode)
        y = torch.tensor([1 if label == "unsafe" else 0 for label in labels], dtype=torch.long)
        encoded = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        steps = max(len(loader) * self.epochs, 1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, steps // 10), num_training_steps=steps)
        self.model.train()
        for _ in range(self.epochs):
            for input_ids, attention_mask, batch_y in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_y).loss
                loss.backward()
                optimizer.step()
                scheduler.step()

    def predict_scores(self, rows: list[dict], mode: str) -> list[float]:
        from torch.utils.data import DataLoader, TensorDataset

        torch = self.torch
        texts = pair_texts(rows, mode)
        encoded = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        scores: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                logits = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device)).logits
                prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
                scores.extend(float(x) for x in prob)
        return scores


def select_threshold(labels: list[str], scores: list[float]) -> TrainResult:
    true = np.asarray([1 if label == "unsafe" else 0 for label in labels], dtype=int)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (np.asarray(scores) >= threshold).astype(int)
        score = f1_score(true, pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return TrainResult(threshold=best_threshold, metrics={"macro_f1": best_f1}, scores=list(scores))


def labels_from_scores(scores: list[float], threshold: float) -> list[str]:
    return ["unsafe" if score >= threshold else "safe" for score in scores]
