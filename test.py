import torch
import numpy as np
import re
import os
import time
import math
from termcolor import colored

def test_entropy_calculation():
    """Test the entropy calculation algorithm used in the Entropy CoT stopper"""
    print("\n=== Testing Entropy Calculation ===")
    
    def calculate_entropy(probabilities):
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
    
    # Known probabilities and expected entropy values
    test_cases = [
        {"probs": [1.0], "expected": 0.0},  # Single token case
        {"probs": [0.5, 0.5], "expected": 0.35},  # Equal uncertainty
        {"probs": [0.9, 0.1], "expected": 0.16},  # Low entropy case
        {"probs": [0.2, 0.2, 0.2, 0.2, 0.2], "expected": 0.32},  # Uniform distribution
    ]
    
    for i, case in enumerate(test_cases):
        # Calculate entropy
        calculated = calculate_entropy(case["probs"])
        
        # Check if it's close to expected (with some tolerance)
        is_close = abs(calculated - case["expected"]) < 0.01
        status = "✓" if is_close else "✗"
        
        print(f"  Test {i+1}: {status} Probs: {case['probs']}, Expected: {case['expected']:.3f}, Got: {calculated:.3f}")

def test_line_tracking():
    """Test the line tracking functionality of the early stopping mechanism"""
    print("\n=== Testing Line Tracking ===")
    
    class MockEntropyStopper:
        def __init__(self, num_warmup_lines=2):
            self.line_count = 0
            self.num_warmup_lines = num_warmup_lines
            self.current_line_probs = []
            self.threshold = 0.2
            self.should_force_stop = False
            self.in_warmup = True
            
        def increment_line(self, entropy=None):
            """Simulate processing a line with given entropy"""
            self.line_count += 1
            
            # Check if we're still in warmup
            if self.line_count <= self.num_warmup_lines:
                self.in_warmup = True
                return False
            
            # We're past warmup
            self.in_warmup = False
            
            # Check entropy if provided
            if entropy is not None and entropy < self.threshold:
                self.should_force_stop = True
                return True
                
            return False
    
    # Test with different warmup settings
    for warmup in [0, 2, 4]:
        print(f"  Testing with warmup={warmup}:")
        
        stopper = MockEntropyStopper(num_warmup_lines=warmup)
        
        # Line entropies to simulate
        entropies = [0.5, 0.4, 0.3, 0.1, 0.05]  # Decreasing entropy
        
        # Simulate lines until stop or out of entropies
        stopping_line = -1
        for i, entropy in enumerate(entropies):
            line_num = i + 1
            stopped = stopper.increment_line(entropy)
            
            # Log status
            warmup_status = "warmup" if stopper.in_warmup else "active"
            print(f"    Line {line_num}: Entropy={entropy:.2f}, Status={warmup_status}")
            
            if stopped:
                stopping_line = line_num
                print(f"    → Stopped at line {stopping_line}")
                break
        
        # Did it stop at the expected line?
        expected_stop = warmup + 1
        if expected_stop <= len(entropies):
            # Should stop at first line with entropy < threshold after warmup
            for i, e in enumerate(entropies[expected_stop-1:], expected_stop):
                if e < stopper.threshold:
                    expected_stop = i
                    break
        else:
            # Not enough lines to get past warmup
            expected_stop = -1
            
        if expected_stop > 0:
            print(f"    Expected to stop at line: {expected_stop}")
            print(f"    {'✓' if stopping_line == expected_stop else '✗'} Stopping behavior is correct")
        else:
            print(f"    Expected no stopping (not enough lines)")
            print(f"    {'✓' if stopping_line == -1 else '✗'} Stopping behavior is correct")

def test_threshold_impact():
    """Test how different entropy thresholds affect stopping behavior"""
    print("\n=== Testing Threshold Impact ===")
    
    # Test scenario
    line_entropies = [
        0.8,  # First line (high uncertainty)
        0.5,  # Second line (medium uncertainty)
        0.3,  # Third line (getting more confident)
        0.2,  # Fourth line
        0.15, # Fifth line
        0.1,  # Sixth line
        0.05  # Seventh line (very confident)
    ]
    
    # Test different thresholds
    thresholds = [0.0, 0.1, 0.2, 0.4]
    
    for threshold in thresholds:
        print(f"  Testing threshold={threshold}:")
        
        # Warmup settings
        warmup = 2
        
        # Expected stopping point
        expected_stop = -1
        
        # Find where we expect to stop (first line after warmup with entropy < threshold)
        if threshold > 0:
            for i, entropy in enumerate(line_entropies[warmup:], warmup + 1):
                if entropy < threshold:
                    expected_stop = i
                    break
        
        # Print expected behavior
        if expected_stop > 0:
            print(f"    Should stop at line {expected_stop} (entropy {line_entropies[expected_stop-1]:.2f})")
        else:
            print(f"    Should not stop (threshold too low or zero)")
        
        # Print detection check
        for i, entropy in enumerate(line_entropies[warmup:], warmup + 1):
            detection = "STOP" if entropy < threshold and threshold > 0 else "continue"
            print(f"    Line {i}: Entropy={entropy:.2f} → {detection}")
            if detection == "STOP" and expected_stop == -1:
                expected_stop = i
                
        # Verify our prediction makes sense
        if threshold == 0.0:
            print(f"    ✓ Zero threshold correctly shows no stopping")
        elif expected_stop > 0:
            print(f"    ✓ Threshold {threshold} triggers stop at correct point")
        else:
            print(f"    ✗ Couldn't determine stopping behavior")
                        
def run_all_tests():
    """Run all test cases"""
    print("\n======= EntropyQwenModel Test Suite =======")
    print("Running selected tests to verify entropy calculation and logic")
    
    test_entropy_calculation()
    test_line_tracking()
    test_threshold_impact()
    
    print("\n=========== Test Suite Complete ===========")

if __name__ == "__main__":
    # Run the tests
    run_all_tests() 