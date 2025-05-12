import torch
import math
import time
import re
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationMixin
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import LogitsProcessor

from threading import Event

class EntropyCoTStopper(LogitsProcessor):
    def __init__(self, entropy_threshold: float = 0.4, thinking_tag_token_ids: List[int] = [151668], period_token_id: int = 198, newline_token_id: int = 10):
        self.thinking_tag_token_ids = thinking_tag_token_ids
        self.stop_ids = [period_token_id, newline_token_id]
        self.entropy_threshold = entropy_threshold
        self.last_token_entropy = None
        self.should_force_stop = False
        self.first_think_token_id = thinking_tag_token_ids[0] if thinking_tag_token_ids else None
        
        self.verbose = True

    def update_entropy(self, token_id: int, entropy: Optional[float]):
        """Update the entropy value if the token is a period or newline"""
        # Check if token is period or newline using their token IDs
        if (token_id in self.stop_ids) and entropy is not None:
            self.last_token_entropy = entropy
            # If entropy is below threshold, set flag to force '</think>' token
            if entropy < self.entropy_threshold:
                self.should_force_stop = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Modify scores to force '</think>' token when entropy is low"""
        if self.should_force_stop and self.first_think_token_id is not None and scores[0, self.first_think_token_id] > 0:
            if self.verbose:
                print(f"Stopping despite score: {scores[0, self.first_think_token_id]}")
                max_score = torch.max(scores[0])
                median_score = torch.median(scores[0])
                print(f"Max score: {max_score}, Median score: {median_score}")
                max_token_id = torch.argmax(scores[0])
                print(f"Max token ID: {max_token_id}") 
        
            # Create a tensor of large negative values
            scores.fill_(-float('inf'))
            # Set the score for the first token of '</think>' to a large positive value
            scores[0, self.first_think_token_id] = 100.0
            # Reset the flag after forcing the token
            self.should_force_stop = False
            
        return scores

class EntropyQwenModel(GenerationMixin):
    """
    A wrapper around the Qwen model that adds entropy-based early stopping for chain-of-thought.
    """
    def __init__(
        self, 
        model_name: str,
        entropy_threshold: float = 0.2,
        min_line_tokens: int = 3,
        num_warmup_lines: int = 1,
        cache_dir: str = None,
        **kwargs
    ):
        """
        Initialize the custom Qwen model with entropy-based stopping.
        
        Args:
            model_name: Name of the Qwen model to load
            entropy_threshold: Value below which generation will stop
            min_line_tokens: Minimum tokens in a line to consider for stopping
            num_warmup_lines: Number of initial thinking lines to ignore before applying stopping
            cache_dir: Directory to cache model files
        """
        # Load model and tokenizer
        if not model_name:
            raise ValueError("model_name is required")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            **kwargs
        )
        
        # Store parameters
        self.entropy_threshold = entropy_threshold
        self.min_line_tokens = min_line_tokens
        self.num_warmup_lines = num_warmup_lines
        
        # Set up proper thinking tag
        self.thinking_tag = "</think>"
        self.thinking_tag_token_ids = self.tokenizer.encode(self.thinking_tag, add_special_tokens=False)
        
        # Get period and newline token IDs
        # Use the first token ID if encoding returns multiple tokens
        period_tokens = self.tokenizer.encode(".", add_special_tokens=False)
        newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        self.period_token_id = period_tokens[0] if period_tokens else 198  # Fallback value
        self.newline_token_id = newline_tokens[0] if newline_tokens else 10  # Fallback value
        
        # Create entropy stopper
        self.entropy_stopper = EntropyCoTStopper(
            entropy_threshold=self.entropy_threshold,
            thinking_tag_token_ids=self.thinking_tag_token_ids,
            period_token_id=self.period_token_id,
            newline_token_id=self.newline_token_id
        )
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self
    
    @property
    def device(self):
        """Get model device"""
        return self.model.device
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        enable_entropy_stopping: bool = True,
        **kwargs
    ) -> Union[torch.LongTensor, CausalLMOutputWithPast]:
        """
        Generate text with entropy-based early stopping during chain-of-thought thinking.
        
        Args:
            input_ids: Token IDs to use as prompt
            enable_entropy_stopping: Whether to use entropy-based stopping
            **kwargs: Additional arguments to pass to model.generate()
        """
        if enable_entropy_stopping:
            # Set up logits processor for entropy-based stopping
            logits_processor = kwargs.get("logits_processor", [])
            if not isinstance(logits_processor, list):
                logits_processor = []
            logits_processor.append(self.entropy_stopper)
            kwargs["logits_processor"] = logits_processor
            
        # Call the model's generate method
        return self.model.generate(input_ids, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Forward pass to the underlying model"""
        return self.model(*args, **kwargs)
    
    def apply_chat_template(self, messages, **kwargs):
        """Apply chat template using the tokenizer"""
        return self.tokenizer.apply_chat_template(messages, **kwargs)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
    
    def _reorder_cache(self, *args, **kwargs):
        """Reorder cache"""
        return self.model._reorder_cache(*args, **kwargs)
    
    def calculate_entropy(self, probabilities: List[float]) -> float:
        """
        Calculate normalized entropy of a sequence of probabilities.
        H(X) = -1/(log(n)) * Σ(p(x_i) * log(p(x_i))) where n is the number of tokens
        """
        if not probabilities:
            return 0.0

        n = len(probabilities)
        if n <= 1:
            return 0.0  # Entropy is 0 for a single token

        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-20

        # Calculate the sum of p * log(p)
        entropy_sum = sum(p * math.log(p + epsilon) for p in probabilities)

        # Normalize by log(n) to get a value between 0 and 1
        normalized_entropy = -1.0 * entropy_sum / (n + epsilon)

        return normalized_entropy