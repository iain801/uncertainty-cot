from typing import List, Optional, Callable


class ProbabilityTracker:
    """
    Utility class that tracks token probabilities and entropy values for visualization.
    Used by EntropyCoTStopper for monitoring and debugging.
    """
    def __init__(self):
        self.token_probs = []
        self.current_line_probs = []
        self.thinking_line_number = 0
        self.line_entropies = []
        self.line_terminations = []  # Stores the terminator type for each line
        self.last_line_real_entropy = 0.0  # Store the actual entropy from the model
        self._calculate_entropy_fn = None  # Will be set by EntropyQwenModel
    
    def add_probability(self, prob):
        """Add a token probability to the tracking lists"""
        self.token_probs.append(prob)
        self.current_line_probs.append(prob)
    
    def new_line(self, terminator_type=None):
        """Process the end of a line, storing entropy and termination info"""
        if self.current_line_probs:
            # Calculate entropy for visualization purposes only if we have a calculation function
            visual_entropy = 0.0
            if self._calculate_entropy_fn:
                visual_entropy = self._calculate_entropy_fn(self.current_line_probs)
            
            # Use the actual entropy from the model's internal calculation if available
            # Otherwise, fall back to the visualization entropy
            displayed_entropy = self.last_line_real_entropy if hasattr(self, 'last_line_real_entropy') else visual_entropy
            
            self.line_entropies.append(displayed_entropy)
            self.line_terminations.append(terminator_type or "unknown")
            
            # Increment our line counter
            self.thinking_line_number += 1
                
            self.current_line_probs = []
    
    def set_real_entropy(self, entropy):
        """Store the actual entropy calculated by the model"""
        self.last_line_real_entropy = entropy
    
    def set_calculate_entropy_fn(self, fn):
        """Set the function to calculate entropy"""
        self._calculate_entropy_fn = fn
    
    def get_line_count(self):
        """Get the current thinking line count"""
        return self.thinking_line_number 

    def reset(self):
        """Reset tracker state for a new generation"""
        self.token_probs = []
        self.current_line_probs = []
        self.thinking_line_number = 0
        self.line_entropies = []
        self.line_terminations = []
        self.last_line_real_entropy = 0.0 