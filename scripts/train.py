import sys
import importlib
import os
from pathlib import Path
from textwrap import dedent

# Ensure the project root (containing the `src` package) is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_MODULES = {
    "torch": "pip install torch --index-url https://download.pytorch.org/whl/cpu",
    "torchaudio": "pip install torchaudio --index-url https://download.pytorch.org/whl/cpu",
    "transformers": "pip install transformers",
    "datasets": "pip install datasets",
    "librosa": "pip install librosa",
    "numpy": "pip install numpy",
    "tqdm": "pip install tqdm",
    "peft": "pip install peft",
    "evaluate": "pip install evaluate",
    "accelerate": "pip install accelerate",
    "soundfile": "pip install soundfile",
    "sentencepiece": "pip install sentencepiece",
}


def check_required_modules():
    missing = []
    for module, hint in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append((module, hint))
    if missing:
        bullet_lines = "\n".join(f"  - {module}: {hint}" for module, hint in missing)
        message = (
            "Missing required dependencies:\n"
            f"{bullet_lines}\n\n"
            "Install them via `pip install -r requirements.txt` or use the hints above."
        )
        raise ModuleNotFoundError(message)


check_required_modules()

from transformers import HfArgumentParser
from src.config import (
    ModelArguments,
    DataArguments,
    WhisperProcessorConfig,
    WhisperTrainingArguments,
    WhisperPredictionArguments,
)
from src.model.whisper_model import load_whisper_model
from src.data.dataset import ChineseTaiwaneseDataset
from src.data.data_collator import WhisperDataCollator
from src.trainers.whisper_trainer import get_trainer
import torch
from src.utils.mlflow_logging import mlflow_logging
from src.utils.logging import logger

# from peft import LoraConfig, TaskType


@mlflow_logging("Whisper_Experiment", "lora")
def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            WhisperTrainingArguments,
            WhisperProcessorConfig,
            WhisperPredictionArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, procrssor_args, prediction_args = (
            parser.parse_json_file(json_file=sys.argv[1])
        )
    else:
        model_args, data_args, training_args, procrssor_args, prediction_args = (
            parser.parse_args_into_dataclasses()
        )

    # Mixed precision guard: fp16 requires CUDA, fall back otherwise.
    device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    if training_args.fp16 and device_type != "cuda":
        logger.warning(
            "Disabling fp16 mixed precision because the current device '%s' does not support CUDA.",
            device_type,
        )
        training_args.fp16 = False
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    if training_args.gradient_checkpointing and device_type != "cuda":
        logger.warning(
            "Disabling gradient checkpointing on device '%s' (only worthwhile on CUDA).",
            device_type,
        )
        training_args.gradient_checkpointing = False
    # Configure LoRA if specified
    peft_config = None
    if model_args.use_peft:
        if model_args.peft_method.lower() == "lora":
            peft_config = {
                "task_type": None,
                "r": model_args.lora_r,
                "lora_alpha": model_args.lora_alpha,
                "lora_dropout": model_args.lora_dropout,
                "bias": "none",
            }
        else:
            raise ValueError(f"Unsupported PEFT method: {model_args.peft_method}")

    compute_dtype = torch.float16 if training_args.fp16 else torch.float32
    model, processor = load_whisper_model(
        model_args.model_name_or_path,
        use_peft=model_args.use_peft,
        peft_config=peft_config,
        language=model_args.language,
        compute_dtype=compute_dtype,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
    )

    # processor.tokenizer.model_max_length = model.config.max_length
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    train_dataset_list = data_args.dataset
    train_dataset, eval_dataset = (
        ChineseTaiwaneseDataset.create_train_and_test_datasets(
            data_args,
            processor,
        )
    )
    logger.info(
        "Finished preprocessing datasets: train=%d samples, eval=%d samples",
        len(train_dataset),
        len(eval_dataset),
    )

    data_collator = WhisperDataCollator(
        processor=processor,
    )
    model.config.use_cache = False
    processor.tokenizer.predict_timestamps = data_args.timestamp

    logger.info("Initialising trainer...")
    trainer = get_trainer(
        model=model,
        args=training_args,
        processor_args=procrssor_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processor=processor,
    )

    logger.info("Starting training loop...")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    # prediction_args = prediction_args.to_dict()
    # prediction_args["eos_token_id"] = [
    #     processor.tokenizer.eos_token_id
    # ] + processor.tokenizer.additional_special_tokens_ids
    # prediction_args["pad_token_id"] = processor.tokenizer.pad_token_id
    # prediction_args["predict_with_generate"] = False
    # metrics = trainer.evaluate(metric_key_prefix="eval", **prediction_args)
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    # trainer.save_state()
    return {
        'checkpoint_dir': training_args.output_dir,
        'base_model_name': model_args.model_name_or_path,
        'data_config': data_args.__dict__,
        "train_dataset": train_dataset_list
    }


if __name__ == "__main__":
    main()
