import torch
import math
from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import LogitsProcessor


class EntropyCoTStopper(LogitsProcessor):
    def __init__(self, entropy_threshold: float = 0.4, thinking_tag_token_ids: List[int] = [151668], terminator_token_ids: List[int] = None, num_warmup_lines: int = 1):
        self.thinking_tag_token_ids = thinking_tag_token_ids
        self.terminator_token_ids = terminator_token_ids or []
        self.entropy_threshold = entropy_threshold
        self.last_token_entropy = None
        self.should_force_stop = False
        self.first_think_token_id = thinking_tag_token_ids[0] if thinking_tag_token_ids else None
        
        self.num_warmup_lines = num_warmup_lines
        self.line_count = 0
        self.current_line_probs = []
        
        self.verbose = True
        self.last_token_id = None
        self.in_thinking_state = True  # Track if we're still in thinking state
        self.already_forced_stop = False  # Track if we've already forced a stop
        self.prob_tracker = None  # Will be set by the model

    def update_entropy(self, token_id: int, entropy: Optional[float]):
        """Update the entropy value if the token is a statement terminator"""
        # Check if token is a statement terminator
        if token_id in self.terminator_token_ids and entropy is not None and self.in_thinking_state:
            self.last_token_entropy = entropy
            
            # Share the real entropy with the ProbabilityTracker if available
            if self.prob_tracker is not None:
                self.prob_tracker.set_real_entropy(entropy)
            
            # Print debug information about the comparison
            if self.verbose:
                print(f"DEBUG: Checking entropy {entropy:.6f} against threshold {self.entropy_threshold:.6f}")
                # Use a small epsilon for floating point comparison
                epsilon = 1e-6
                is_below = entropy < (self.entropy_threshold - epsilon)
                print(f"DEBUG: Is entropy < threshold? {is_below} (with epsilon={epsilon})")
                print(f"DEBUG: Line count: {self.line_count}, warmup needed: {self.num_warmup_lines}")
                print(f"DEBUG: Past warmup? {self.line_count >= self.num_warmup_lines}")
                print(f"DEBUG: Already forced stop? {self.already_forced_stop}")
                print(f"DEBUG: Should trigger stop? {is_below and self.line_count >= self.num_warmup_lines and not self.already_forced_stop}")
            
            # Use a small epsilon for floating point comparison
            epsilon = 1e-6
            if entropy < (self.entropy_threshold - epsilon):
                if self.line_count >= self.num_warmup_lines and not self.already_forced_stop:
                    self.should_force_stop = True
                    self.already_forced_stop = True  # Set flag to prevent multiple stops
                    if self.verbose:
                        print(f"Low entropy detected: {entropy:.6f} (threshold: {self.entropy_threshold:.6f})")
                else:
                    if self.verbose and not self.already_forced_stop:
                        print(f"Not stopping - line count: {self.line_count}, warmup needed: {self.num_warmup_lines}")
            
            # Note: We no longer increment line_count here because it's now done in __call__

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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits and handle entropy calculation and stopping.
        This method is called for each token generated.
        """
        # Check if we've already generated the think token
        if self.last_token_id == self.first_think_token_id and self.in_thinking_state:
            if self.verbose:
                print("Found </think> token, exiting thinking state")
            self.in_thinking_state = False
            self.should_force_stop = False
            return scores
        
        # Skip entropy checks if we're not in thinking state
        if not self.in_thinking_state:
            return scores
            
        # Get the probabilities from scores using softmax
        probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # Get the token with highest probability
        next_token_id = torch.argmax(scores[0]).item()
        next_token_prob = probs[0, next_token_id].item()
        
        # Track probabilities for current line
        self.current_line_probs.append(next_token_prob)
        
        # Check if the last token was a terminator and update line_count
        # print(f"id: {self.last_token_id}")
        if self.last_token_id in self.terminator_token_ids:
            terminator_type = self.terminator_id_to_type.get(self.last_token_id, "unknown")

            self.line_count += 1
            if self.verbose:
                print(f"DEBUG: Found terminator {terminator_type} (ID: {self.last_token_id}), incrementing line count to {self.line_count}")
        
        # Calculate current entropy for debugging and share with tracker
        if self.prob_tracker is not None:
            if len(self.current_line_probs) > 1:
                current_entropy = self.calculate_entropy(self.current_line_probs)
                
                # Share the current real-time entropy with the ProbabilityTracker
                self.prob_tracker.set_real_entropy(current_entropy)
                    
                if self.verbose:
                    print(f"DEBUG: Current line entropy: {current_entropy:.6f} (tokens: {len(self.current_line_probs)}, line: {self.line_count})")
        
        # Check if the last token was a terminator and if so, calculate entropy and update
        if self.last_token_id in self.terminator_token_ids:
            entropy = self.calculate_entropy(self.current_line_probs)
            self.update_entropy(self.last_token_id, entropy)
            
            # Reset current line probabilities for next line
            self.current_line_probs = []
        
        # Check if we need to stop thinking based on entropy
        if self.should_force_stop and self.first_think_token_id is not None:
            if self.verbose:
                print(f"Forcing </think> token (ID: {self.first_think_token_id})")
                print(f"Current max token would be: {next_token_id} with score {scores[0, next_token_id]}")
            
            # Create a tensor of large negative values
            scores.fill_(-float('inf'))
            # Set the score for the first token of '</think>' to a large positive value
            scores[0, self.first_think_token_id] = 100.0
            # Reset the flag after forcing the token to prevent multiple stops
            self.should_force_stop = False
        
        # Store the current token ID for next iteration
        self.last_token_id = next_token_id
            
        return scores

class EntropyQwenModel(GenerationMixin):
    """
    A wrapper around the Qwen model that adds entropy-based early stopping for chain-of-thought.
    """
    def __init__(
        self, 
        model_name: str,
        entropy_threshold: float = 0.2,
        num_warmup_lines: int = 1,
        statement_terminators: List[str] = None,
        cache_dir: str = None,
        verbose: bool = False,  # Default to False for cleaner output
        **kwargs
    ):
        """
        Initialize the custom Qwen model with entropy-based stopping.
        
        Args:
            model_name: Name of the Qwen model to load
            entropy_threshold: Value below which generation will stop
            num_warmup_lines: Number of initial thinking lines to ignore before applying stopping
            statement_terminators: List of strings that denote the end of a statement (default: [".", "\n"])
            cache_dir: Directory to cache model files
            verbose: Whether to print detailed debug information
        """
        # Load model and tokenizer
        if not model_name:
            raise ValueError("model_name is required")
        
        # Store verbose setting
        self.verbose = verbose
        
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
        self.num_warmup_lines = num_warmup_lines
        
        # Set default statement terminators if none provided
        self.statement_terminators = statement_terminators or [".", "\n"]
        
        # Set up proper thinking tag
        self.thinking_tag = "</think>"
        self.thinking_tag_token_ids = self.tokenizer.encode(self.thinking_tag, add_special_tokens=False)
        
        # Get token IDs for statement terminators
        self.terminator_token_ids = []
        self.terminator_id_to_type = {}  # Map from token ID to terminator type
        
        if self.verbose:
            print(f"Setting up statement terminators: {self.statement_terminators}")
            
        for terminator in self.statement_terminators:
            token_ids = self.tokenizer.encode(terminator, add_special_tokens=False)
            if token_ids:
                # Use the first token ID if encoding returns multiple tokens
                token_id = token_ids[0]
                self.terminator_token_ids.append(token_id)
                self.terminator_id_to_type[token_id] = terminator
                if self.verbose:
                    print(f"  Terminator '{terminator}' encoded to token ID: {token_id}")
                    # Decode the token to verify
                    decoded = self.tokenizer.decode([token_id])
                    print(f"    Decodes back to: '{decoded}'")
            elif self.verbose:
                print(f"  Warning: Terminator '{terminator}' could not be encoded to token IDs")
        
        print(f"terminator_token_ids: {self.terminator_token_ids}")
        
        # Create entropy stopper
        self.entropy_stopper = EntropyCoTStopper(
            entropy_threshold=self.entropy_threshold,
            thinking_tag_token_ids=self.thinking_tag_token_ids,
            terminator_token_ids=self.terminator_token_ids,
            num_warmup_lines=self.num_warmup_lines
        )
        
        # Set verbose mode from model
        self.entropy_stopper.verbose = self.verbose
        
        # Pass the terminator_id_to_type mapping to the stopper
        self.entropy_stopper.terminator_id_to_type = self.terminator_id_to_type
        
        # Set up logits processor for entropy-based stopping
        if entropy_threshold > 0:
            logits_processor = kwargs.get("logits_processor", [])
            if not isinstance(logits_processor, list):
                logits_processor = []
            logits_processor.append(self.entropy_stopper)
            kwargs["logits_processor"] = logits_processor
            
        torch.compile(self.model)
    
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
        **kwargs
    ) -> Union[torch.LongTensor, CausalLMOutputWithPast]:
        """
        Generate text with entropy-based early stopping during chain-of-thought thinking.
        
        Args:
            input_ids: Token IDs to use as prompt
            enable_entropy_stopping: Whether to use entropy-based stopping
            **kwargs: Additional arguments to pass to model.generate()
        """
            
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
        Delegates to the entropy_stopper's method.
        """
        return self.entropy_stopper.calculate_entropy(probabilities)

    def set_prob_tracker(self, prob_tracker):
        """Set the probability tracker for accessing real entropy values"""
        self.entropy_stopper.prob_tracker = prob_tracker