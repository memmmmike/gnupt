"""
GPT-2 based scoring for AI text detection.

Provides perplexity and log-likelihood calculations used by multiple detectors.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Optional


class GPT2Scorer:
    """
    GPT-2 model for computing perplexity and log-likelihood.

    Used by both GPT2Detector (perplexity-based) and DetectGPT (perturbation-based).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_id: str = "gpt2",
        stride: int = 512
    ):
        """
        Initialize GPT-2 scorer.

        Args:
            device: Compute device ("cuda", "cpu", or None for auto-detect)
            model_id: HuggingFace model ID ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
            stride: Stride for sliding window (smaller = more accurate but slower)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_id = model_id
        self.stride = stride

        print(f"Loading GPT-2 model '{model_id}' on {device}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.model.eval()
        print(f"GPT-2 model '{model_id}' loaded successfully!")

        self.max_length = self.model.config.n_positions

    def get_perplexity(self, text: str) -> float:
        """
        Calculate perplexity for text.

        Lower perplexity = more predictable = more likely AI-generated.

        Args:
            text: Text to analyze

        Returns:
            Perplexity score (integer)
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return float(ppl.cpu().numpy())

    def get_log_likelihood(self, text: str) -> float:
        """
        Calculate log-likelihood for text.

        Used by DetectGPT for comparing original vs perturbations.

        Args:
            text: Text to analyze

        Returns:
            Log-likelihood score (negative value, higher = more likely)
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        log_likelihood = -1 * torch.stack(nlls).sum() / end_loc
        return float(log_likelihood.cpu().numpy())


# Singleton instances for different model sizes
_scorers: dict[str, GPT2Scorer] = {}


def get_scorer(model_id: str = "gpt2", device: Optional[str] = None) -> GPT2Scorer:
    """
    Get or create a GPT2Scorer instance.

    Uses singleton pattern to avoid loading the same model multiple times.

    Args:
        model_id: Model identifier
        device: Compute device

    Returns:
        GPT2Scorer instance
    """
    key = f"{model_id}:{device}"
    if key not in _scorers:
        _scorers[key] = GPT2Scorer(device=device, model_id=model_id)
    return _scorers[key]
