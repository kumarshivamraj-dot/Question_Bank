from __future__ import annotations

import hashlib
import math
from collections import Counter

from study_pipeline.text_utils import tokenize


class EmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self.dimensions

        counts = Counter(tokens)
        vector = [0.0] * self.dimensions
        for token, count in counts.items():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dimensions
            sign = -1.0 if int(digest[-1], 16) % 2 else 1.0
            vector[index] += sign * float(count)
        return normalize_vector(vector)


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, text: str) -> list[float]:
        import requests

        payload = {"model": self.model, "input": text}
        response = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=120,
        )
        if response.status_code == 404:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()
        if "embeddings" in data and data["embeddings"]:
            return normalize_vector(data["embeddings"][0])
        if "embedding" in data:
            return normalize_vector(data["embedding"])
        raise ValueError("Ollama embedding response did not contain a vector")


def normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
