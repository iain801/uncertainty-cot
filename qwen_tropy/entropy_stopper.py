import numpy as np
import torch
from typing import List, Optional
from transformers.generation import LogitsProcessor


class EntropyCoTStopper(LogitsProcessor):
    """
    A LogitsProcessor that can detect when a language model's thinking should be stopped
    based on a low entropy threshold.
    """
    def __init__(self, entropy_threshold: float = 0.4, thinking_tag_token_ids: List[int] = [151668], 
                 terminator_token_ids: List[int] = None, num_warmup_lines: int = 1):
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
        
        # Vectorized calculation using numpy for better performance
        probs = np.array(probabilities)
        entropy_sum = np.sum(probs * np.log2(probs + epsilon))
        
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

    def reset(self):
        """Reset the state variables relevant to stopping between generations"""
        self.last_token_entropy = None
        self.should_force_stop = False
        self.line_count = 0
        self.current_line_probs = []
        self.last_token_id = None
        self.in_thinking_state = True
        self.already_forced_stop = False
        
        if self.verbose:
            print("DEBUG: EntropyCoTStopper has been reset for a new generation")
