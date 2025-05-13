import torch
import time
import sys
import random
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from qwen_tropy import EntropyQwenModel
import datasets

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(420)
np.random.seed(42)

model_name = "Qwen/Qwen3-0.6B"
stop_threshold = 0  # set to 0 to disable stopping
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


def run_benchmark(dataset, num_samples=None, output_file="benchmark_results.parquet"):
    """
    Run the benchmark on the provided dataset
    
    Args:
        dataset: The dataset to benchmark
        num_samples: Number of samples to process (None for all)
        output_file: Path to save the results
        
    Returns:
        DataFrame with benchmark results
    """
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
        
        # Extract answer from \boxed{} tags
        boxed_pattern = r'\\boxed{([^}]*)}'
        boxed_match = re.findall(boxed_pattern, response_content)
        model_answer = boxed_match[-1].strip() if boxed_match else response_content.strip()
        
        # Calculate token statistics
        thinking_tokens = len(tokenizer.encode(thinking_content)) if thinking_content else 0
        response_tokens = len(tokenizer.encode(response_content)) if response_content else 0
        total_tokens = thinking_tokens + response_tokens
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        
        # Determine if the answer is correct
        is_correct = model_answer.lower() == sample["answer"].lower()
        
        # Store the results
        results.append({
            "problem_id": sample.get("id", f"sample_{len(results)}"),
            "problem": sample["problem"],
            "generated_answer": model_answer,
            "correct_answer": sample["answer"],
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
    # Load dataset
    print("Loading MATH-500 dataset...")
    dataset = datasets.load_dataset(
        path="HuggingFaceH4/MATH-500",
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
    
    # Generate output filename with timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"results/benchmark_{model_name.replace('/', '_')}_{timestamp}.parquet"
    
    # Run benchmark
    run_benchmark(dataset, num_samples=num_samples, output_file=output_file)


if __name__ == "__main__":
    main() 