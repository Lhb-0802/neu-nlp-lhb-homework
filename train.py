from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)
from transformers import LlamaForCausalLM, LlamaTokenizer

def train(
        # model/data params
        base_model: str = "/data0/luohaibo/llama-weights/llama-2-7b/",
        data_path: str = "/data0/luohaibo/llm/processed_data/train_dev.json",  # may be modified
        output_dir: str = "/data0/luohaibo/llm/output/",
        micro_batch_size: int = 2,
        gradient_accumulation_steps: int = 2,
        num_epochs: int = 6,
        learning_rate: float = 1e-5,
        val_set_size: int = 1000,

        # lora hyperparams
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ]
):
    device_map = "auto"

    # Step 1: Load the data
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # Step 2: Load the model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True, # Add this for using int8
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
        # max_seq_length=512
    )
    tokenizer.pad_token_id = 0
    # Add this for training LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    # model = prepare_model_for_int8_training(model)  # Add this for using int8

    # Step 3: Tokenize the data
    def tokenize(data):
        source_ids = tokenizer.encode(data['input'])
        target_ids = tokenizer.encode(data['output'])

        input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    # split the data to train/val set
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=False, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(tokenize)
    )
    val_data = (
        train_val["test"].shuffle().map(tokenize)

    )
    # Step 4: Initiate the trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            # weight_decay=0.01,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            # fp16=True,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=2000,
            save_steps=2000,
            output_dir=output_dir,
            save_total_limit=5,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train()
    # Step 5: save the model
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
