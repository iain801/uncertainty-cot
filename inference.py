import torch
import time
import sys
import random
import numpy as np
import math
import re
from transformers import TextIteratorStreamer
from termcolor import colored
from threading import Thread, Event
from qwen_tropy import EntropyQwenModel

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def get_probability_color(prob):
    """Return a color based on token probability"""
    if prob > 0.9:
        return "green"
    elif prob > 0.7:
        return "blue"
    elif prob > 0.5:
        return "yellow"
    elif prob > 0.3:
        return "magenta"
    else:
        return "red"

def calculate_entropy(probabilities):
    """Calculate normalized entropy of a sequence of probabilities
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

def is_empty_line(line):
    """Check if a line is empty or contains only whitespace"""
    return not line or line.isspace()

def split_into_sentences(text):
    """Split text into sentences by periods and newlines
    
    Handles cases like decimals, ellipses, etc. to avoid splitting incorrectly
    """
    # First split by newlines
    lines = text.split('\n')
    sentences = []
    
    for line in lines:
        if not line.strip():
            sentences.append(line)
            continue
            
        # Split by periods, but be careful with decimals, abbreviations, etc.
        # This regex splits by periods that are followed by a space or end of string
        # and not preceded by common abbreviations
        fragments = re.split(r'(?<!\s[A-Z])(?<!\d)(?<!\.)\.(?=\s|$)', line)
        
        for i, fragment in enumerate(fragments):
            if i < len(fragments) - 1:
                # Add the period back for all but the last fragment
                sentences.append(fragment + '.')
            else:
                sentences.append(fragment)
    
    return sentences

# Initialize model
model_name = "Qwen/Qwen3-0.6B"
print(f"Loading model {model_name}...")

# Create EntropyQwenModel with entropy-based early stopping
model = EntropyQwenModel(
    model_name=model_name,
    cache_dir="tmp/",
    entropy_threshold=0.4,  # Entropy threshold for early stopping
    min_line_tokens=3,      # Minimum tokens in a line to consider
    num_warmup_lines=1      # Number of lines to ignore before stopping
)

# Access the tokenizer from the model
tokenizer = model.tokenizer

# prepare the model input
prompt = "If a doctor gives you 3 pills and tells you to take one pill every half hour, how long would it last before you've taken all the pills?"
messages = [
    {"role": "user", "content": prompt}
]
text = model.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion with streaming
print("[Generation Start]")
start_time = time.time()

# Create proper streamer - don't skip special tokens so we can find </think>
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

# Signal for when generation is complete
generation_done = Event()

# Custom tracker for visualization
class ProbabilityTracker:
    def __init__(self):
        self.token_probs = []
        self.current_line_probs = []
        self.thinking_line_number = 0
    
    def add_probability(self, prob):
        self.token_probs.append(prob)
        self.current_line_probs.append(prob)
    
    def new_line(self):
        if self.current_line_probs:
            self.thinking_line_number += 1
            self.current_line_probs = []

# Create tracker
prob_tracker = ProbabilityTracker()

# Function to run in thread with proper error handling
def generate_with_streamer():
    try:
        # Use the model's generate method directly with the streamer
        output = model.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            do_sample=True,
            top_p=0.95,    
            temperature=0.6,
            top_k=20,
            streamer=streamer
        )
        print(f"Generation completed with {len(output[0])} tokens")
    except Exception as e:
        print(f"Error in generation: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal that generation is done
        generation_done.set()

# Generate in a separate thread that doesn't block program exit
generation_thread = Thread(target=generate_with_streamer)
generation_thread.daemon = True
generation_thread.start()

# Collect the full text output and print each chunk as it's generated
generated_text = ""
is_thinking = True
current_line = ""
current_buffer = ""  # Buffer to accumulate text for sentence splitting
manually_stopped = False
current_prob = 0.5  # Default probability for visualization

try:
    for text_chunk in streamer:
        if text_chunk is None:
            continue
            
        generated_text += text_chunk
        current_buffer += text_chunk
        
        # This is just for visualization - actual early stopping is handled by EntropyQwenModel
        current_prob = 0.5 + (0.5 * random.random())  # Simulated probability for visualization
        prob_tracker.add_probability(current_prob)
        prob_color = get_probability_color(current_prob)
        
        # Handle line tracking during thinking - process on periods and newlines
        if is_thinking:
            # Check if we have a potential line ending (period or newline)
            if '.' in current_buffer or '\n' in current_buffer:
                # Split into sentences and process each one
                sentences = split_into_sentences(current_buffer)
                
                if len(sentences) > 1:  # We have complete sentences
                    # Process all complete sentences except the last one (may be incomplete)
                    for i, sentence in enumerate(sentences[:-1]):
                        # For empty sentences, just print a newline without entropy info
                        if is_empty_line(sentence):
                            print()
                            continue
                        
                        # Calculate line entropy for display
                        line_entropy = calculate_entropy(prob_tracker.current_line_probs)
                        
                        # Display sentence with entropy info
                        line_info = f" [Line: {prob_tracker.thinking_line_number}, Entropy: {line_entropy:.4f}]"
                        print(colored(sentence, prob_color), end="", flush=True)
                        print(colored(line_info, "cyan"), end="\n", flush=True)
                        
                        # Track new line
                        prob_tracker.new_line()
                    
                    # Keep the last (potentially incomplete) sentence in the buffer
                    current_buffer = sentences[-1]
        
        # Check if we've switched from thinking to response
        if is_thinking and "</think>" in text_chunk:
            is_thinking = False
            # Print any remaining buffer content
            if current_buffer.strip():
                print(colored(current_buffer, prob_color), end="", flush=True)
                current_buffer = ""
            
            # Print the chunk in two colors - colored for thinking part, default for response part
            thinking_part, response_part = text_chunk.split("</think>", 1)
            if thinking_part:
                print(colored(thinking_part, prob_color), end="", flush=True)
            print("</think>" + response_part, end="", flush=True)
        else:
            # For regular chunk printing that doesn't involve complete sentences
            if is_thinking and '.' not in text_chunk and '\n' not in text_chunk:
                # Just print the chunk directly if it doesn't contain line endings
                print(colored(text_chunk, prob_color), end="", flush=True)
                current_buffer = ""  # Already printed
            elif not is_thinking and not manually_stopped:
                # Print in default color (not colored) for final answer
                print(text_chunk, end="", flush=True)
        
        # Flush to ensure immediate output
        sys.stdout.flush()
        
except Exception as e:
    print(f"\nError during streaming: {str(e)}")
    import traceback
    traceback.print_exc()

# Wait for generation to complete with a timeout
print("\nWaiting for generation to complete...")
generation_done.wait(timeout=60)  # Wait up to 60 seconds
generation_time = time.time() - start_time
print("\n[Generation Done]")

# Find the thinking and response portions using the </think> tag
THINK_END_TAG = "</think>"
think_end_pos = generated_text.find(THINK_END_TAG)

if think_end_pos != -1:
    thinking_content = generated_text[:think_end_pos].strip()
    content = generated_text[think_end_pos + len(THINK_END_TAG):].strip()
else:
    # No thinking tag found
    thinking_content = ""
    content = generated_text.strip()

# Calculate token statistics
thinking_tokens = len(tokenizer.encode(thinking_content)) if thinking_content else 0
total_tokens = len(tokenizer.encode(generated_text))
thinking_percentage = (thinking_tokens / total_tokens * 100) if total_tokens > 0 else 0
tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

print("\n\n--- Performance Metrics ---")
print(f"Generation time: {generation_time:.2f} seconds")
print(f"Thinking tokens: {thinking_tokens}")
print(f"Total tokens: {total_tokens}")
print(f"Thinking percentage: {thinking_percentage:.2f}%")
print(f"Generation speed: {tokens_per_second:.2f} tokens/second")

# Display probability statistics if available
if prob_tracker.token_probs:
    avg_prob = sum(prob_tracker.token_probs) / len(prob_tracker.token_probs)
    min_prob = min(prob_tracker.token_probs)
    max_prob = max(prob_tracker.token_probs)
    print(f"\nToken probability stats - Avg: {avg_prob:.4f}, Min: {min_prob:.4f}, Max: {max_prob:.4f}")
