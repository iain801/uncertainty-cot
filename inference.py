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
import matplotlib.pyplot as plt

import datasets

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(420)
np.random.seed(42)

model_name = "Qwen/Qwen3-0.6B"
stop_threshold = 0.175
show_stream = False

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

# Custom tracker for visualization
class ProbabilityTracker:
    def __init__(self):
        self.token_probs = []
        self.current_line_probs = []
        self.thinking_line_number = 0
        self.line_entropies = []
        self.line_terminations = []  # 'period' or 'newline'
    
    def add_probability(self, prob):
        self.token_probs.append(prob)
        self.current_line_probs.append(prob)
    
    def new_line(self, termination_type='newline'):
        if self.current_line_probs:
            entropy = model.calculate_entropy(self.current_line_probs)
            self.line_entropies.append(entropy)
            self.line_terminations.append(termination_type)
            
            # Update the entropy stopper with the current line's entropy
            # Use the actual token IDs from the model
            token_id = model.period_token_id if termination_type == 'period' else model.newline_token_id
            model.entropy_stopper.update_entropy(token_id, entropy)
            
            self.thinking_line_number += 1
            self.current_line_probs = []

def main():
    global model  # Make model available to ProbabilityTracker
    
    # Initialize model
    print(f"Loading model {model_name}...")
    model = EntropyQwenModel(
        model_name=model_name,
        cache_dir="tmp/",
        entropy_threshold=stop_threshold,  # Entropy threshold for early stopping
        min_line_tokens=3,      # Minimum tokens in a line to consider
        num_warmup_lines=10,      # Number of lines to ignore before stopping
    )

    # Access the tokenizer from the model
    tokenizer = model.tokenizer

    # prepare the model input
    dataset = datasets.load_dataset(
        path="HuggingFaceH4/MATH-500",
        split="test",
        cache_dir="tmp/",
        num_proc=10
    )
    sample = random.choice(dataset)
    prompt = sample["problem"]
    
    # prompt = "If a doctor gives you 3 pills and tells you to take one pill every half hour, how long would it last before you've taken all the pills?"
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

    # Create tracker
    prob_tracker = ProbabilityTracker()

    # Choose different generation approaches based on streaming preference
    if show_stream:
        # Create proper streamer - don't skip special tokens so we can find </think>
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

        # Signal for when generation is complete
        generation_done = Event()

        # Function to run in thread with proper error handling
        def generate_with_streamer():
            try:
                # Use the model's generate method directly with the streamer
                output = model.generate(
                    **model_inputs,
                    max_new_tokens=32768,
                    do_sample=True,
                    top_p=0.95,    
                    temperature=0.6,
                    top_k=20,
                    streamer=streamer,
                    enable_entropy_stopping=False
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

        print("[Streaming generation...]")

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
                                    prob_tracker.new_line('newline')
                                    continue
                                
                                # Calculate line entropy for display
                                line_entropy = model.calculate_entropy(prob_tracker.current_line_probs)
                                
                                # Determine termination type
                                termination_type = 'period' if sentence.strip().endswith('.') else 'newline'
                                
                                # Display sentence with entropy info
                                line_info = f" [Line: {prob_tracker.thinking_line_number}, Entropy: {line_entropy:.4f}]"
                                print(colored(sentence, prob_color), end="", flush=True)
                                print(colored(line_info, "cyan"), end="\n", flush=True)
                                
                                # Track new line
                                prob_tracker.new_line(termination_type)
                            
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
                    if response_part:
                        print(response_part, end="", flush=True)
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
    
    else:
        # Non-streaming approach - simpler and more direct
        print("[Generation in progress (streaming disabled)...]")
        
        try:
            # Direct generation without streaming
            output = model.generate(
                **model_inputs,
                max_new_tokens=32768,
                do_sample=True,
                top_p=0.95,    
                temperature=0.6,
                top_k=20,
                enable_entropy_stopping=False
            )
            
            # Decode the generated tokens
            generated_text = tokenizer.decode(output[0][len(model_inputs.input_ids[0]):], skip_special_tokens=False)
            
            # Calculate probabilities for metrics (simplified since we're not streaming)
            for _ in range(len(tokenizer.encode(generated_text))):
                current_prob = 0.5 + (0.5 * random.random())  # Simulated probability
                prob_tracker.add_probability(current_prob)
                
            generation_time = time.time() - start_time
            print(f"\n[Generation Done] Completed with {len(output[0])} tokens in {generation_time:.2f} seconds")
            
            # Print the question and answer
            print("\n--- Question ---")
            print(prompt)
            
            # Find the answer after </think> tag
            THINK_END_TAG = "</think>"
            think_end_pos = generated_text.find(THINK_END_TAG)
            
            print("\n--- Generated Answer ---")
            if think_end_pos != -1:
                answer = generated_text[think_end_pos + len(THINK_END_TAG):].strip()
                print(answer)
            else:
                print(generated_text)
                
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            generation_time = time.time() - start_time
    
    print(f"\n--- Correct Answer ---\n{sample['answer']}")

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

    # Plot entropy values with markers for termination type
    if prob_tracker.line_entropies:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(prob_tracker.line_entropies) + 1), prob_tracker.line_entropies, '-b')
        
        # Add markers for termination types
        for i, (entropy, term_type) in enumerate(zip(prob_tracker.line_entropies, prob_tracker.line_terminations)):
            if term_type == 'period':
                plt.plot(i + 1, entropy, 'ko')  # black dot for period
            else:
                plt.plot(i + 1, entropy, 'rx')  # red x for newline
        
        plt.axhline(y=stop_threshold, color='green', linestyle='--', label='Stopping Threshold')
        
        plt.xlabel('Reasoning Step')
        plt.ylabel('Entropy')
        plt.title('Entropy at Each Reasoning Step')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot
        plt.savefig('visualization_entropy.png')
        print("Entropy plot saved as 'visualization_entropy.png'")

if __name__ == "__main__":
    main()
