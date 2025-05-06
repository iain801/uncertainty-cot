import torch
import math
import time
import re
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationMixin, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast
from threading import Event


class EntropyBasedStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on the entropy of generated tokens in a sequence.
    Only applies during the thinking phase (before </think> tag).
    Considers both periods and newlines as line separators.
    """
    def __init__(
        self, 
        entropy_threshold: float = 0.4, 
        min_line_tokens: int = 3,
        num_warmup_lines: int = 1,
        thinking_tag: str = "</think>",
        tokenizer = None
    ):
        """
        Initialize the entropy-based stopping criteria.
        
        Args:
            entropy_threshold: Value below which the generation will stop
            min_line_tokens: Minimum number of tokens in a line to consider for stopping
            num_warmup_lines: Number of initial thinking lines to ignore before applying stopping criteria
            thinking_tag: The tag that marks the end of thinking
            tokenizer: The tokenizer to use (needed to detect newlines)
        """
        self.entropy_threshold = entropy_threshold
        self.min_line_tokens = min_line_tokens
        self.num_warmup_lines = num_warmup_lines
        self.thinking_tag = thinking_tag
        self.tokenizer = tokenizer
        
        # State tracking
        self.token_probs = []
        self.current_line_probs = []
        self.current_line_text = ""
        self.thinking_line_number = 0
        self.last_was_newline = False
        self.last_was_period = False
        self.should_stop = False
        self.stop_reason = ""
        self.thinking_phase = True  # Track if we're still in thinking phase
        self.last_token_ids = []    # Keep track of recent tokens for period detection
        
        # Map newline token IDs
        self.newline_token_ids = []
        if tokenizer:
            self.newline_token_ids = [
                tokenizer.encode("\n", add_special_tokens=False)[0],
                tokenizer.encode("\r", add_special_tokens=False)[0] if len(tokenizer.encode("\r", add_special_tokens=False)) > 0 else -1
            ]
            
        # Get period token ID
        self.period_token_id = None
        if tokenizer:
            period_tokens = tokenizer.encode(".", add_special_tokens=False)
            if period_tokens:
                self.period_token_id = period_tokens[0]
                
        # Get </think> token IDs for checking end of thinking phase
        self.thinking_end_token_ids = []
        if tokenizer:
            self.thinking_end_token_ids = tokenizer.encode(thinking_tag, add_special_tokens=False)
    
    def is_empty_line(self, text):
        """Check if a line is empty or contains only whitespace"""
        return not text or text.isspace()
    
    def is_end_of_sentence(self, token_id, text):
        """
        Check if this token marks the end of a sentence (period)
        Excludes common cases like decimals, abbreviations, etc.
        """
        if token_id != self.period_token_id:
            return False
            
        # Avoid treating periods in numbers or common abbreviations as sentence endings
        if not text or len(text) < 3:
            return False
            
        # Check the last few characters to see if this is a true sentence ending
        # Periods followed by space or at the end are likely sentence endings
        if text[-1] == '.':
            # Check not decimal number
            if text[-2].isdigit():
                return False
                
            # Check not common abbreviation (e.g., "Dr.", "Mr.")
            if len(text) >= 3 and text[-3:] in ['.Dr', '.Mr', '.Ms']:
                return False
                
            return True
            
        return False
    
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
        epsilon = 1e-10
        
        # Calculate the sum of p * log(p)
        entropy_sum = sum(p * math.log(p + epsilon) for p in probabilities)
        
        # Normalize by log(n) to get a value between 0 and 1
        normalized_entropy = -1.0 * entropy_sum / (math.log(n) + epsilon)
        
        return normalized_entropy
    
    def is_line_end(self, token_id: int) -> bool:
        """Check if the token is a line ending token (newline)"""
        return token_id in self.newline_token_ids
    
    def has_thinking_ended(self, input_ids: torch.LongTensor) -> bool:
        """Check if the </think> tag has been generated"""
        if not self.thinking_end_token_ids or len(self.thinking_end_token_ids) == 0:
            return False
            
        # Get the sequence length to check against
        seq_len = input_ids.shape[1]
        
        # If there aren't enough tokens yet to contain the tag, it hasn't ended
        if seq_len < len(self.thinking_end_token_ids):
            return False
            
        # Check if the last N tokens match the </think> tag
        last_tokens = input_ids[0, -len(self.thinking_end_token_ids):].tolist()
        return last_tokens == self.thinking_end_token_ids
    
    def process_new_line(self):
        """Process a completed line and decide whether to stop"""
        # Skip empty or whitespace-only lines
        if self.is_empty_line(self.current_line_text):
            self.current_line_probs = []  # Reset probabilities for new line
            self.current_line_text = ""   # Reset text for new line
            return
        
        # Calculate entropy for the current line
        line_entropy = self.calculate_entropy(self.current_line_probs)
        
        # Only consider stopping if we have enough tokens and have passed the warmup lines
        if len(self.current_line_probs) >= self.min_line_tokens and self.thinking_line_number >= self.num_warmup_lines:
            if line_entropy < self.entropy_threshold:
                self.should_stop = True
                self.stop_reason = f"entropy={line_entropy:.4f}"
        
        # Reset line tracking
        self.current_line_probs = []
        self.current_line_text = ""
        self.thinking_line_number += 1
    
    def add_token_prob(self, prob: float, token_id: int = None):
        """Add a token probability and process line endings"""
        self.token_probs.append(prob)
        self.current_line_probs.append(prob)
        
        # Add token to current line text if tokenizer is available
        if token_id is not None and self.tokenizer:
            token_text = self.tokenizer.decode([token_id])
            self.current_line_text += token_text
            
            # Keep track of last tokens
            self.last_token_ids.append(token_id)
            if len(self.last_token_ids) > 5:  # Keep a rolling window
                self.last_token_ids.pop(0)
        
        # Check for line endings (newlines)
        if token_id is not None and self.is_line_end(token_id):
            if not self.last_was_newline:  # Handle consecutive newlines
                self.process_new_line()
            self.last_was_newline = True
            self.last_was_period = False
        # Check for sentence endings (periods)
        elif token_id is not None and token_id == self.period_token_id:
            # Check if it's a true sentence ending period (not decimal, abbreviation, etc.)
            if self.is_end_of_sentence(token_id, self.current_line_text):
                if not self.last_was_period:  # Handle consecutive periods
                    self.process_new_line()
                self.last_was_period = True
                self.last_was_newline = False
            else:
                self.last_was_period = False
                self.last_was_newline = False
        else:
            self.last_was_newline = False
            self.last_was_period = False
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if generation should stop based on entropy.
        Only applies during thinking phase (before </think> tag).
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
            scores: Tensor of shape (batch_size, vocabulary_size)
                Scores for each possible next token
        
        Returns:
            bool: True if generation should stop, False otherwise
        """
        # First, check if we've exited the thinking phase
        if not self.thinking_phase:
            return False  # Never stop after thinking phase
            
        # Check if the thinking phase has just ended
        if self.has_thinking_ended(input_ids):
            self.thinking_phase = False
            return False  # Don't stop at the </think> tag itself
        
        # Get the last token ID
        last_token_id = input_ids[0, -1].item()
        
        # Get probability of the most likely token that was selected
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(scores[0], dim=-1)
        
        # Get the maximum probability (most likely token)
        max_prob = torch.max(probs).item()
        
        # Add token probability and maybe process line ending
        self.add_token_prob(max_prob, last_token_id)
        
        # Return stopping decision - only applies during thinking phase
        return self.should_stop


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
        # Create stopping criteria if enabled
        if enable_entropy_stopping:
            entropy_stopping = EntropyBasedStoppingCriteria(
                entropy_threshold=self.entropy_threshold,
                min_line_tokens=self.min_line_tokens,
                num_warmup_lines=self.num_warmup_lines,
                tokenizer=self.tokenizer
            )
            
            # Add to existing stopping criteria or create new list
            stopping_criteria = kwargs.get('stopping_criteria', StoppingCriteriaList())
            if not isinstance(stopping_criteria, StoppingCriteriaList):
                stopping_criteria = StoppingCriteriaList([stopping_criteria])
            
            stopping_criteria.append(entropy_stopping)
            kwargs['stopping_criteria'] = stopping_criteria
            
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