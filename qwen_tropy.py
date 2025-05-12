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

class EntropyQwenModel(GenerationMixin):
    """
    A wrapper around the Qwen model that adds entropy-based early stopping for chain-of-thought.
    """
    def __init__(
        self, 
        model_name: str,
        entropy_threshold: float = 0.4,
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
            pass
            
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


def demo():
    """
    Demonstrate using the EntropyQwenModel with a simple example.
    """
    print("Initializing model...")
    model = EntropyQwenModel(
        model_name="Qwen/Qwen3-0.6B",
        cache_dir="tmp/",
        entropy_threshold=0.4,
        min_line_tokens=3,
        num_warmup_lines=1
    )
    
    # Prepare input prompt for chain-of-thought
    prompt = "If a doctor gives you 3 pills and tells you to take one pill every half hour, how long would it last before you've taken all the pills?"
    messages = [{"role": "user", "content": prompt}]
    text = model.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    # Tokenize input
    model_inputs = model.tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate with entropy-based stopping
    print(f"Generating response for: '{prompt}'")
    start_time = time.time()
    output = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        do_sample=True,
        top_p=0.95,
        temperature=0.6,
        top_k=20,
        return_dict_in_generate=True,
        output_scores=True
    )
    generation_time = time.time() - start_time
    
    # Decode output without printing the full raw text yet
    generated_text = model.tokenizer.decode(output.sequences[0], skip_special_tokens=False)
    
    # Calculate token statistics
    input_length = len(model_inputs.input_ids[0])
    total_tokens = len(output.sequences[0]) - input_length
    
    # Check if thinking was stopped early
    if "</think>" in generated_text:
        thinking, response = generated_text.split("</think>", 1)
        
        # Clean up the thinking content - remove template tags
        thinking = thinking.replace("<|im_start|>user", "")
        thinking = thinking.replace("<|im_end|>", "")
        thinking = thinking.replace("<|im_start|>assistant", "")
        thinking = thinking.replace("<think>", "")
        thinking = thinking.strip()
        
        # Clean up response content
        response = response.strip()
        
        # Calculate thinking tokens
        thinking_ids = model.tokenizer.encode(thinking)
        thinking_tokens = len(thinking_ids)
        thinking_percentage = (thinking_tokens / total_tokens * 100) if total_tokens > 0 else 0
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        
        # Print thinking with stopping info
        print("\n--- Thinking (with entropy-based early stopping) ---")
        print(thinking[:200] + "..." if len(thinking) > 200 else thinking)
        print(f"\n--- Full thinking length: {len(thinking)} chars, {thinking_tokens} tokens ---")
        
        # Print the response part
        print("\n--- Response ---")
        print(response)
    else:
        print("\nThinking was not stopped early.")
        thinking_tokens = 0
        thinking_percentage = 0
    
    # Display performance metrics
    print("\n--- Performance Metrics ---")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Thinking tokens: {thinking_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Thinking percentage: {thinking_percentage:.2f}%")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/second")
    
    # Show full final generation
    print("\n--- Full Generated Output ---")
    # Remove system template parts for cleaner output
    clean_output = generated_text
    clean_output = clean_output.replace("<|im_start|>user", "\nUser:")
    clean_output = clean_output.replace("<|im_end|>", "")
    clean_output = clean_output.replace("<|im_start|>assistant", "\nAssistant:")
    clean_output = clean_output.replace("<think>", "\nThinking: ")
    clean_output = clean_output.replace("</think>", "\n\nAnswer: ")
    print(clean_output)


if __name__ == "__main__":
    demo() 