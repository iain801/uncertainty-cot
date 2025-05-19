import torch
import time
import sys
import random
import numpy as np
import re
import pandas as pd
import os
from tqdm import tqdm
from qwen_tropy import EntropyQwenModel
import datasets
from math_verify import verify, parse, LatexExtractionConfig, LatexNormalizationConfig

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(420)
np.random.seed(42)

model_name = "Qwen/Qwen3-0.6B"
stop_threshold = 0.15  # set to 0 to disable stopping
warmup_lines = 4
statement_terminators = [
    "\n", "\n\n", "\n\n\n", "\n\n\n\n", "\n\n\n\n\n", "\n\n\n\n\n\n",
    "\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n\n",
    "\n\n\n\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n\n\n\n\n",
    " \n", " \n\n", " \n\n\n", " \n\n\n\n", " \n\n\n\n\n",
    ".\n", ".\n\n", ".\n\n\n", ".\n\n\n\n", ".\n\n\n\n\n",
    "!\n", "!\n\n", "!\n\n\n", "!\n\n\n\n",
    "?\n", "?\n\n", "?\n\n\n", "?\n\n\n\n",
]


def extract_boxed_content(text):
    """Extract content from \boxed{} expressions, handling nested braces correctly."""
    results = []
    i = 0
    while i < len(text):
        # Find the start of a boxed expression
        boxed_start = text.find("\\boxed{", i)
        if boxed_start == -1:
            break
            
        # Find the matching closing brace
        brace_start = boxed_start + len("\\boxed{")
        brace_level = 1
        brace_end = brace_start
        
        while brace_level > 0 and brace_end < len(text):
            if text[brace_end] == '{':
                brace_level += 1
            elif text[brace_end] == '}':
                brace_level -= 1
            brace_end += 1
        
        if brace_level == 0:
            # Successfully found the matching closing brace
            content = text[brace_start:brace_end-1]
            results.append(content)
            
        # Move past this boxed expression
        i = brace_end
        
    return results


def run_benchmark(dataset, dataset_name, num_samples=None, output_file=None):
    """
    Run the benchmark on the provided dataset
    
    Args:
        dataset: The dataset to benchmark
        dataset_name: Name of the dataset
        num_samples: Number of samples to process (None for all)
        output_file: Path to save the results (if None, a path will be generated)
        
    Returns:
        DataFrame with benchmark results
    """
    # Format the model name to be directory-friendly
    safe_model_name = model_name.replace("/", "_")
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create output directories if they don't exist
    if output_file is None:
        results_dir = f"results/{safe_model_name}/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        output_file = f"{results_dir}/benchmark_{stop_threshold}_{timestamp}.parquet"
    
    print(f"Loading model {model_name}...")
    model = EntropyQwenModel(
        model_name=model_name,
        cache_dir="tmp/",
        entropy_threshold=stop_threshold,
        num_warmup_lines=warmup_lines,
        statement_terminators=statement_terminators,
        verbose=False
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
        
        # Decode the generated text
        generated_text = tokenizer.decode(output[0][len(model_inputs.input_ids[0]):], skip_special_tokens=False)
        
        # Find the thinking and response portions
        THINK_END_TAG = "</think>"
        think_end_pos = generated_text.find(THINK_END_TAG)
        
        if think_end_pos != -1:
            thinking_content = generated_text[:think_end_pos].strip()
            response_content = generated_text[think_end_pos + len(THINK_END_TAG):].strip()
        else:
            thinking_content = ""
            response_content = generated_text.strip()
        
        # Calculate token statistics
        thinking_tokens = len(tokenizer.encode(thinking_content)) if thinking_content else 0
        response_tokens = len(tokenizer.encode(response_content)) if response_content else 0
        total_tokens = thinking_tokens + response_tokens
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        
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
        
        is_correct = verify(
            gold_answer, model_answer
        )
                
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
    # Dataset name and path
    dataset_path = "HuggingFaceH4/MATH-500"
    dataset_name = "MATH-500"
    
    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = datasets.load_dataset(
        path=dataset_path,
        split="test",
        cache_dir="tmp/",
        num_proc=10
    )
    
    # Optional: Parse command line arguments for number of samples
    num_samples = None
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of samples: {sys.argv[1]}")
            sys.exit(1)
    
    # Run benchmark with new directory structure
    run_benchmark(dataset, dataset_name, num_samples=num_samples)


if __name__ == "__main__":
    main() 