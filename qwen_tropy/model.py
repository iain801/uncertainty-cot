import torch
from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .entropy_stopper import EntropyCoTStopper
from .rolling_stopper import RollingEntropyCoTStopper


class EntropyQwenModel(GenerationMixin):
    """
    A wrapper around the Qwen model that adds entropy-based early stopping for chain-of-thought.
    """
    def __init__(
        self, 
        model_name: str,
        entropy_threshold: float = 0.2,
        window_size: int = 3,          # Window size (also used for warmup lines)
        statement_terminators: List[str] = None,
        cache_dir: str = None,
        verbose: bool = False,         # Default to False for cleaner output
        use_rolling_entropy: bool = False,  # Whether to use rolling entropy window
        **kwargs
    ):
        """
        Initialize the custom Qwen model with entropy-based stopping.
        
        Args:
            model_name: Name of the Qwen model to load
            entropy_threshold: Value below which generation will stop
            window_size: Size of rolling window AND number of lines before stopping
            statement_terminators: List of strings that denote the end of a statement (default: [".", "\n"])
            cache_dir: Directory to cache model files
            verbose: Whether to print detailed debug information
            use_rolling_entropy: Whether to use rolling window-based entropy
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
        self.window_size = window_size
        
        # Set up proper thinking tag
        self.thinking_tag = "</think>"
        self.thinking_tag_token_ids = self.tokenizer.encode(self.thinking_tag, add_special_tokens=False)
        
        # Get token IDs for statement terminators
        self.terminator_token_ids = []
        self.terminator_id_to_type = {}  # Map from token ID to terminator type
        
        if self.verbose:
            print(f"Setting up statement terminators:")
            
        if statement_terminators:
            for terminator in statement_terminators:
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
        else:
            self.terminator_token_ids = self.get_all_newline_token_ids()
            if self.verbose:
                for token_id in self.terminator_token_ids:
                    print(f"  Terminator token ID: {token_id}")
                    decoded = self.tokenizer.decode([token_id])
                    print(f"    Decodes back to: '{decoded}'")

        # Create appropriate entropy stopper based on configuration
        if use_rolling_entropy:
            # Use the rolling entropy stopper with the window size
            self.entropy_stopper = RollingEntropyCoTStopper(
                entropy_threshold=self.entropy_threshold,
                thinking_tag_token_ids=self.thinking_tag_token_ids,
                terminator_token_ids=self.terminator_token_ids,
                window_size=window_size
            )
            if self.verbose:
                print(f"Using rolling entropy with window size: {window_size}")
        else:
            # Use the standard entropy stopper
            self.entropy_stopper = EntropyCoTStopper(
                entropy_threshold=self.entropy_threshold,
                thinking_tag_token_ids=self.thinking_tag_token_ids,
                terminator_token_ids=self.terminator_token_ids,
                num_warmup_lines=window_size
            )
            if self.verbose:
                print(f"Using standard entropy with warmup lines: {window_size}")
        
        # Set verbose mode from model
        self.entropy_stopper.verbose = self.verbose
        
        # Pass the terminator_id_to_type mapping to the stopper
        self.entropy_stopper.terminator_id_to_type = self.terminator_id_to_type
            
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
        # Reset the generation state for a fresh start
        self.reset_generation_state()
        
        if self.entropy_threshold > 0:
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
        Delegates to the entropy_stopper's method.
        """
        return self.entropy_stopper.calculate_entropy(probabilities)

    def set_prob_tracker(self, prob_tracker):
        """Set the probability tracker for accessing real entropy values"""
        self.entropy_stopper.prob_tracker = prob_tracker
        
        # Set the calculate_entropy function in the tracker so it can calculate entropy values
        if hasattr(prob_tracker, 'set_calculate_entropy_fn'):
            prob_tracker.set_calculate_entropy_fn(self.calculate_entropy) 
            
    def get_all_newline_token_ids(self):
        """Get all newline token IDs"""
        # Get the complete vocabulary dictionary
        vocab = self.tokenizer.get_vocab()
        newline_tokens = {token: token_id for token, token_id in vocab.items() 
                         if '\n' in token}
        
        return list(newline_tokens.values())

    def reset_generation_state(self):
        """Reset the internal state of the entropy stopper for a new generation"""
        if hasattr(self, 'entropy_stopper') and self.entropy_stopper is not None:
            self.entropy_stopper.reset()
            if self.verbose:
                print(f"Reset entropy stopper state for new generation")
            
        # Also reset any tracker if connected
        if hasattr(self.entropy_stopper, 'prob_tracker') and self.entropy_stopper.prob_tracker is not None:
            # Reset tracking state
            self.entropy_stopper.prob_tracker.reset()
