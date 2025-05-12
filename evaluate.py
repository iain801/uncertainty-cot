from datetime import timedelta

import torch
from transformers import BitsAndBytesConfig, GenerationConfig

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.transformers.transformers_model import TransformersModelConfig, GenerationParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    torch.set_float32_matmul_precision('high')
else:
    accelerator = None

def main():
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
        override_batch_size=1
    )
    
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=0.95,    
        temperature=0.6,
        top_k=20,
    )
    
    generation_parameters = GenerationParameters(
        top_p=0.95,    
        temperature=0.6,
        top_k=20,
    )

    model_name = "Qwen/Qwen3-0.6B"
    model_config = TransformersModelConfig(
        pretrained=model_name,
        accelerator=accelerator, 
        model_parallel=True,
        compile=True,
        device="cuda",
        dtype="auto",
        use_chat_template=True,
        generation_parameters=generation_parameters,
    )

    task = "leaderboard|hellaswag|10|1,leaderboard|gsm8k|3|1"
    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    main()



