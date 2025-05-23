import torch
import time
import random
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from qwen_tropy import EntropyQwenModel
import datasets
from math_verify import verify, parse, LatexExtractionConfig, LatexNormalizationConfig

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Default values (will be overridden by command-line arguments)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_STOP_THRESHOLD = 0.0  # set to 0 to disable stopping
DEFAULT_WARMUP_LINES = 10
DEFAULT_DATASET_PATH = "HuggingFaceH4/MATH-500"

statement_terminators = [
    "\n", "\n\n", "\n\n\n", "\n\n\n\n", "\n\n\n\n\n", "\n\n\n\n\n\n",
    "\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n\n",
    "\n\n\n\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n\n\n\n\n",
    " \n", " \n\n", " \n\n\n", " \n\n\n\n", " \n\n\n\n\n",
    ".\n", ".\n\n", ".\n\n\n", ".\n\n\n\n", ".\n\n\n\n\n",
    "!\n", "!\n\n", "!\n\n\n", "!\n\n\n\n",
    "?\n", "?\n\n", "?\n\n\n", "?\n\n\n\n",
]

def run_benchmark(dataset, dataset_name, model_name, stop_threshold, window_size, 
                 num_samples=None, output_file=None, verbose=False, 
                 use_rolling_entropy=False):
    """
    Run the benchmark on the provided dataset
    
    Args:
        dataset: The dataset to benchmark
        dataset_name: Name of the dataset
        model_name: Name of the model to use
        stop_threshold: Entropy threshold for early stopping
        window_size: Window size and warmup lines
        num_samples: Number of samples to process (None for all)
        output_file: Path to save the results (if None, a path will be generated)
        verbose: Whether to print detailed logs
        use_rolling_entropy: Whether to use rolling window entropy
        
    Returns:
        DataFrame with benchmark results
    """
    # Format the model name to be directory-friendly
    safe_model_name = model_name.replace("/", "_")
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create output directories if they don't exist
    if output_file is None:
        # Add entropy details to output path
        entropy_suffix = f"rolling{window_size}" if use_rolling_entropy else f"warmup{window_size}"
        results_dir = f"results/{safe_model_name}/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        output_file = f"{results_dir}/benchmark_{stop_threshold}_{entropy_suffix}_{timestamp}.parquet"
    
    print(f"Loading model {model_name}...")
    
    model = EntropyQwenModel(
        model_name=model_name,
        cache_dir="tmp/",
        entropy_threshold=stop_threshold,
        window_size=window_size,
        statement_terminators=statement_terminators,
        verbose=verbose,
        use_rolling_entropy=use_rolling_entropy
    )
    tokenizer = model.tokenizer
    
    # Limit the number of samples if requested
    if num_samples is not None and num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Results storage
    results = []
    
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
    
    # Process each sample with progress bar
    for sample in tqdm(dataset, desc="Processing samples"):
        start_time = time.time()
        
        # Prepare prompt
        prompt = "Solve the following problem, seperating each step with a newline, then provide the solution in within a single \\boxed{} statement: \n" + sample["problem"]
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
        
        # Generate response
        output = model.generate(
            **model_inputs,
            max_new_tokens=32768,
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            top_k=20
        )
        
        generation_time = time.time() - start_time
        
        # Get the generated token IDs
        generated_token_ids = output[0][len(model_inputs.input_ids[0]):]
        
        # Find the thinking and response portions using token ID
        THINK_END_TOKEN = 151668
        try:
            think_end_pos = generated_token_ids.tolist().index(THINK_END_TOKEN)
            thinking_tokens = think_end_pos
            response_tokens = len(generated_token_ids) - think_end_pos - 1
        except ValueError:
            thinking_tokens = 0
            response_tokens = len(generated_token_ids)
            
        total_tokens = len(generated_token_ids)
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        
        # Split content based on token position
        if think_end_pos != -1:
            response_content = tokenizer.decode(generated_token_ids[think_end_pos + 1:]).strip()
        else:
            response_content = generated_text.strip()
        
        try:
            model_answer = parse(response_content, [latex_config])
        except Exception as e:
            print(f"Error parsing answer: {e}")
            model_answer = response_content
            
        try:
            gold_answer = parse(f"\\boxed{{{sample['answer']}}}", [latex_config])
        except Exception as e:
            print(f"Error parsing gold answer: {e}")
            gold_answer = sample["answer"]
        
        try:
            is_correct = verify(
                gold_answer, model_answer
            )
        except Exception as e:
            print(f"Error during verification: {e}")
            is_correct = False
                
        # Store the results
        results.append({
            "problem_id": sample.get("id", f"sample_{len(results)}"),
            "problem": sample["problem"],
            "generated_answer": str(model_answer[0]) if isinstance(model_answer, list) and len(model_answer) > 0 else str(model_answer),
            "correct_answer": str(gold_answer[0]) if isinstance(gold_answer, list) and len(gold_answer) > 0 else str(gold_answer),
            "is_correct": is_correct,
            "thinking_tokens": thinking_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "generation_time_seconds": generation_time,
            "tokens_per_second": tokens_per_second
        })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to parquet
    df.to_parquet(output_file)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    print("\nBenchmark Summary:")
    print(f"Total samples processed: {len(df)}")
    print(f"Correct answers: {df['is_correct'].sum()} ({df['is_correct'].mean() * 100:.2f}%)")
    print(f"Average thinking tokens: {df['thinking_tokens'].mean():.2f}")
    print(f"Average response tokens: {df['response_tokens'].mean():.2f}")
    print(f"Average total tokens: {df['total_tokens'].mean():.2f}")
    print(f"Average generation time: {df['generation_time_seconds'].mean():.2f} seconds")
    print(f"Average tokens per second: {df['tokens_per_second'].mean():.2f}")
    
    return df


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run benchmarks on models with entropy-based early stopping")
    
    # Add arguments
    parser.add_argument(
        "--model", 
        "-m",
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--threshold", 
        "-t",
        type=float, 
        default=DEFAULT_STOP_THRESHOLD,
        help=f"Entropy threshold for early stopping (default: {DEFAULT_STOP_THRESHOLD}, 0 to disable)"
    )
    parser.add_argument(
        "--window", 
        "-w",
        type=int, 
        default=DEFAULT_WARMUP_LINES,
        help=f"Window size and warmup lines (default: {DEFAULT_WARMUP_LINES})"
    )
    parser.add_argument(
        "--dataset-path", 
        "-d",
        type=str, 
        default=DEFAULT_DATASET_PATH,
        help=f"Path to dataset (default: {DEFAULT_DATASET_PATH})"
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default=None,
        help=f"Name of dataset (default: derived from dataset path)"
    )
    parser.add_argument(
        "--samples", 
        "-n",
        type=int, 
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output", 
        "-o",
        type=str, 
        default=None,
        help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--verbose", 
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--rolling", 
        "-r",
        action="store_true",
        help="Use rolling average entropy"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # If dataset name not provided, derive it from the path
    if args.dataset_name is None:
        args.dataset_name = args.dataset_path.split('/')[-1]
    
    # Load dataset
    print(f"Loading dataset {args.dataset_name} from {args.dataset_path}...")
    dataset = datasets.load_dataset(
        path=args.dataset_path,
        split="test",
        cache_dir="tmp/",
        num_proc=10
    )
    
    # Run benchmark with parsed arguments
    run_benchmark(
        dataset=dataset,
        dataset_name=args.dataset_name,
        model_name=args.model,
        stop_threshold=args.threshold,
        window_size=args.window,
        num_samples=args.samples,
        output_file=args.output,
        verbose=args.verbose,
        use_rolling_entropy=args.rolling
    )


if __name__ == "__main__":
    main() 