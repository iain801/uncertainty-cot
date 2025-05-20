from typing import List, Optional
from .entropy_stopper import EntropyCoTStopper

class RollingEntropyCoTStopper(EntropyCoTStopper):
    """Entropy-based CoT stopper that uses a rolling average window for more stable stopping decisions"""
    
    def __init__(self, entropy_threshold: float = 0.4, thinking_tag_token_ids: List[int] = [151668], 
                 terminator_token_ids: List[int] = None, window_size: int = 3):
        # Initialize the parent class with window size as warmup lines
        # Warmup should equal window size to ensure the window is filled
        super().__init__(
            entropy_threshold=entropy_threshold,
            thinking_tag_token_ids=thinking_tag_token_ids,
            terminator_token_ids=terminator_token_ids,
            num_warmup_lines=window_size
        )
        
        # Set window size and initialize the entropy window
        self.window_size = window_size
        self.entropy_window = []
        
    def update_entropy(self, token_id: int, entropy: Optional[float]):
        """Update the entropy value and check if we should stop based on rolling average"""
        # Check if token is a statement terminator and we're still in thinking state
        if token_id in self.terminator_token_ids and entropy is not None and self.in_thinking_state:
            # Store the current entropy in the window
            self.entropy_window.append(entropy)
            
            # Keep only the most recent window_size values
            if len(self.entropy_window) > self.window_size:
                self.entropy_window.pop(0)
                
            # Calculate rolling average
            if len(self.entropy_window) > 0:
                rolling_avg = sum(self.entropy_window) / len(self.entropy_window)
                
                # Share the real entropy with the ProbabilityTracker if available
                if self.prob_tracker is not None:
                    self.prob_tracker.set_real_entropy(rolling_avg)
                
                # Store the rolling average as the last entropy
                self.last_token_entropy = rolling_avg
                
                # Print debug information
                if self.verbose:
                    print(f"DEBUG: Window: {self.entropy_window}")
                    print(f"DEBUG: Rolling average entropy: {rolling_avg:.6f}")
                    print(f"DEBUG: Checking rolling avg {rolling_avg:.6f} against threshold {self.entropy_threshold:.6f}")
                    print(f"DEBUG: Line count: {self.line_count}, window size: {self.window_size}")
                    print(f"DEBUG: Window filled? {len(self.entropy_window) >= self.window_size}")
                
                # Check if rolling average is below threshold and window is filled
                epsilon = 1e-6
                if rolling_avg < (self.entropy_threshold - epsilon):
                    if self.line_count >= self.window_size and not self.already_forced_stop:
                        self.should_force_stop = True
                        self.already_forced_stop = True
                        if self.verbose:
                            print(f"Low rolling average entropy detected: {rolling_avg:.6f} (threshold: {self.entropy_threshold:.6f})")
                    else:
                        if self.verbose and not self.already_forced_stop:
                            print(f"Not stopping - line count: {self.line_count}, window needed: {self.window_size}") 

    def reset(self):
        """Reset the state variables relevant to stopping between generations"""
        # Reset parent class variables
        super().reset()
        
        # Reset rolling entropy window
        self.entropy_window = []
        
        if self.verbose:
            print("DEBUG: RollingEntropyCoTStopper's entropy window has been reset") 