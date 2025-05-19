import torch
import time
import sys
import random
import numpy as np
import re
from transformers import TextIteratorStreamer
from termcolor import colored
from threading import Thread, Event
from qwen_tropy import EntropyQwenModel
import matplotlib.pyplot as plt
from math_verify import verify, parse, LatexExtractionConfig, LatexNormalizationConfig

import datasets

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

model_name = "Qwen/Qwen3-0.6B"
stop_threshold = 0.00001 # set to 0 to disable stopping
warmup_lines = 4
show_stream = True


statement_terminators = [
    "\n","\n\n","\n\n\n","\n\n\n\n","\n\n\n\n\n","\n\n\n\n\n\n",
    "\n\n\n\n\n\n\n","\n\n\n\n\n\n\n\n","\n\n\n\n\n\n\n\n\n",
    "\n\n\n\n\n\n\n\n\n\n","\n\n\n\n\n\n\n\n\n\n\n","\n\n\n\n\n\n\n\n\n\n\n\n",
    " \n", " \n\n", " \n\n\n"," \n\n\n\n"," \n\n\n\n\n",
    ".\n",".\n\n",".\n\n\n",".\n\n\n\n",".\n\n\n\n\n",
    "!\n", "!\n\n", "!\n\n\n","!\n\n\n\n",
    "?\n", "?\n\n", "?\n\n\n","?\n\n\n\n",
]

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
    """Split text into sentences by statement terminators
    
    Handles cases like decimals, ellipses, etc. to avoid splitting incorrectly
    """
    # First split by newlines if newline is in the terminators
    if "\n" in statement_terminators:
        lines = text.split('\n')
    else:
        lines = [text]  # Just treat the whole text as one line
        
    sentences = []
    
    for line in lines:
        if not line.strip():
            sentences.append(line)
            continue
        
        # Start with the line as current text
        current = line
        
        # Process each terminator in the list (except newline which was handled above)
        for terminator in [t for t in statement_terminators if t != "\n"]:
            if terminator in current:
                if terminator == ".":
                    # Special handling for periods to avoid splitting decimals, etc.
                    fragments = re.split(r'(?<!\s[A-Z])(?<!\d)(?<!\.)\.(?=\s|$)', current)
                    processed = []
                    
                    for i, fragment in enumerate(fragments):
                        if i < len(fragments) - 1:
                            # Add the period back for all but the last fragment
                            processed.append(fragment + '.')
                        else:
                            processed.append(fragment)
                    
                    current = "\n".join(processed)  # Join with newlines to split again
                else:
                    # Simple splitting for other terminators
                    parts = current.split(terminator)
                    processed = []
                    
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1:
                            # Add the terminator back for all but the last part
                            processed.append(part + terminator)
                        else:
                            processed.append(part)
                    
                    current = "\n".join(processed)
        
        # Split by the newlines we added and add to sentences
        sentences.extend([s for s in current.split("\n") if s])
    
    return sentences

# Custom tracker for visualization
class ProbabilityTracker:
    def __init__(self):
        self.token_probs = []
        self.current_line_probs = []
        self.thinking_line_number = 0
        self.line_entropies = []
        self.line_terminations = []  # Stores the terminator type for each line
        self.last_line_real_entropy = 0.0  # Store the actual entropy from the model
    
    def add_probability(self, prob):
        self.token_probs.append(prob)
        self.current_line_probs.append(prob)
    
    def new_line(self, terminator_type=None):
        if self.current_line_probs:
            # Calculate entropy for visualization purposes only
            visual_entropy = model.calculate_entropy(self.current_line_probs)
            
            # Use the actual entropy from the model's internal calculation if available
            # Otherwise, fall back to the visualization entropy
            displayed_entropy = self.last_line_real_entropy if hasattr(self, 'last_line_real_entropy') else visual_entropy
            
            self.line_entropies.append(displayed_entropy)
            self.line_terminations.append(terminator_type or "unknown")
            
            # Get the actual line number from the entropy stopper if possible
            if hasattr(model, 'entropy_stopper') and hasattr(model.entropy_stopper, 'line_count'):
                self.thinking_line_number = model.entropy_stopper.line_count 
            else:
                # Fallback to our own counting (legacy)
                self.thinking_line_number += 1
                
            self.current_line_probs = []
    
    def set_real_entropy(self, entropy):
        """Store the actual entropy calculated by the model"""
        self.last_line_real_entropy = entropy

def main():
    global model  # Make model available to ProbabilityTracker
    
    # Initialize model
    print(f"Loading model {model_name}...")
    model = EntropyQwenModel(
        model_name=model_name,
        cache_dir="tmp/",
        entropy_threshold=stop_threshold,  # Entropy threshold for early stopping
        num_warmup_lines=warmup_lines,    # Number of lines to ignore before stopping
        statement_terminators=statement_terminators,  # Pass the statement terminators
        verbose=False  # Set to True for detailed debugging output
    )

    # Access the tokenizer from the model
    tokenizer = model.tokenizer

    # Configure Math-Verify for answer verification
    normalization_config = LatexNormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=True,
        nits=True,
        boxed="all",
        equations=True
    )
    
    # Configure Math-Verify for latex extraction
    latex_config = LatexExtractionConfig(    
        try_extract_without_anchor=True,
        boxed_match_priority=0,
        normalization_config=normalization_config
    )

    # prepare the model input
    dataset = datasets.load_dataset(
        path="HuggingFaceH4/MATH-500",
        split="test",
        cache_dir="tmp/",
        num_proc=10
    )
    sample = random.choice(dataset)
    prompt = "Solve the following problem, seperating each step with a newline, then provide the solution in within a single \\boxed{} statement: \n" + sample["problem"]
    
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
    final_response = ""
    response_content = ""
    
    # Connect the tracker to the model
    model.set_prob_tracker(prob_tracker)

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

        print("[Streaming generation...]")

        # Collect the full text output and print each chunk as it's generated
        generated_text = ""
        is_thinking = True
        current_line = ""
        current_buffer = ""  # Buffer to accumulate text for sentence splitting
        manually_stopped = False
        current_prob = 0.5  # Default probability for visualization
        response_text = ""  # Track the response part separately

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
                
                # Handle line tracking during thinking - process on terminators
                if is_thinking:
                    # Check if we have a potential line ending (any terminator)
                    has_terminator = any(term in current_buffer for term in statement_terminators)
                    if has_terminator:
                        # Split into sentences and process each one
                        sentences = split_into_sentences(current_buffer)
                        
                        if len(sentences) > 1:  # We have complete sentences
                            # Process all complete sentences except the last one (may be incomplete)
                            for i, sentence in enumerate(sentences[:-1]):
                                # For empty sentences, just print a newline without entropy info
                                if is_empty_line(sentence):
                                    print()
                                    prob_tracker.new_line("\n")
                                    continue
                                
                                # Calculate line entropy for display
                                line_entropy = model.calculate_entropy(prob_tracker.current_line_probs)
                                
                                # Use the actual entropy from the model's internal calculation if available
                                displayed_entropy = prob_tracker.last_line_real_entropy if hasattr(prob_tracker, 'last_line_real_entropy') else line_entropy
                                
                                # Determine termination type based on statement terminators
                                terminator_type = None
                                sentence_stripped = sentence.strip()
                                for terminator in statement_terminators:
                                    if sentence_stripped.endswith(terminator):
                                        terminator_type = terminator
                                        break
                                # Default to newline if no terminator found
                                if terminator_type is None:
                                    terminator_type = "\n"
                                
                                # Display sentence with entropy info
                                line_info = f" [Line: {prob_tracker.thinking_line_number}, Entropy: {displayed_entropy:.4f}, Tokens: {len(prob_tracker.current_line_probs)}]"
                                print(colored(sentence, prob_color), end="", flush=True)
                                print(colored(line_info, "cyan"), end="\n", flush=True)
                                
                                # Track new line
                                prob_tracker.new_line(terminator_type)
                            
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
                        # Start collecting the response text
                        response_text += response_part
                elif not is_thinking:
                    # For regular chunk printing that doesn't involve complete sentences
                    # Print in default color (not colored) for final answer
                    print(text_chunk, end="", flush=True)
                    # Continue collecting the response text
                    response_text += text_chunk
                elif not manually_stopped and not has_terminator:
                    # Print the chunk directly if it doesn't contain line endings
                    print(colored(text_chunk, prob_color), end="", flush=True)
                    current_buffer = ""  # Already printed
                
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
        
        # Set final_response to the accumulated response text
        final_response = response_text
        response_content = response_text
    
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
                top_k=20
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
                final_response = generated_text[think_end_pos + len(THINK_END_TAG):].strip()
                response_content = final_response
                print(final_response)
            else:
                final_response = generated_text
                response_content = final_response
                print(final_response)
                
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            generation_time = time.time() - start_time
    
    # Extract answer from \boxed{} tags
    print("\n--- Answer Verification ---")
    
    # Parse and verify the model answer and ground truth using Math-Verify
    try:
        model_answer = parse(response_content, [latex_config])
        if model_answer:
            print(f"Parsed model answer: {model_answer[0]}")
        else:
            print("Failed to parse model answer")
            model_answer = response_content
    except Exception as e:
        print(f"Error parsing answer: {e}")
        model_answer = response_content
        
    try:
        gold_answer = parse(f"\\boxed{{{sample['answer']}}}", [latex_config])
        if gold_answer:
            print(f"Parsed gold answer: {gold_answer[0]}")
        else:
            print("Failed to parse gold answer")
            gold_answer = sample["answer"]
    except Exception as e:
        print(f"Error parsing gold answer: {e}")
        gold_answer = sample["answer"]
    
    try:
        is_correct = verify(gold_answer, model_answer)
        print(f"Answer is correct: {is_correct}")
    except Exception as e:
        print(f"Error during verification: {e}")
        is_correct = False
    
    # Extract boxed answer for display
    boxed_pattern = r'\\boxed{([^}]*)}'
    boxed_match = re.findall(boxed_pattern, final_response)
    final_answer = boxed_match[-1].strip() if boxed_match else final_response.strip()
    
    # Log whether we found a boxed answer
    if boxed_match:
        print(f"\n--- Extracted Boxed Answer ---")
    else:
        print(f"\n--- No Boxed Answer Found, Using Full Response ---")
    
    print(f"{final_answer}")
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
    print(f"Response tokens: {total_tokens - thinking_tokens}")
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
        
        # Define marker styles dynamically based on terminators
        # Use a set of default markers that will be assigned to terminators
        default_markers = ["ko", "rx", "g^", "bs", "mD", "cP", "yh"]
        marker_styles = {"unknown": "k*"}  # Default marker for unknown terminator
        
        # Assign markers to each terminator
        for i, terminator in enumerate(statement_terminators):
            marker_idx = min(i, len(default_markers) - 1)  # Avoid index out of range
            marker_styles[terminator] = default_markers[marker_idx]
        
        # Add markers for termination types
        for i, (entropy, terminator) in enumerate(zip(prob_tracker.line_entropies, prob_tracker.line_terminations)):
            # Get marker style for this terminator (or default to unknown marker)
            marker = marker_styles.get(terminator, marker_styles["unknown"])
            plt.plot(i + 1, entropy, marker)
        
        # Add legend for marker styles
        for terminator, marker in marker_styles.items():
            # Replace newline with \n for display
            label = f"'{terminator.replace('\n', '\\n')}'"
            if terminator == "unknown":
                label = "unknown"
            plt.plot([], [], marker, label=label)
        
        plt.axhline(y=stop_threshold, color='green', linestyle='--', label='Stopping Threshold')
        
        plt.xlabel('Reasoning Step')
        plt.ylabel('Entropy')
        plt.title('Entropy at Each Reasoning Step')
        plt.grid(True, linestyle='--', alpha=0.7)
        # plt.legend()
        
        # Save the plot
        plt.savefig('visualization_entropy.png')
        print("Entropy plot saved as 'visualization_entropy.png'")

if __name__ == "__main__":
    main()
