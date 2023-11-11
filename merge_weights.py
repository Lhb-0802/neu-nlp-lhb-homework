from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

base_model_name_or_path = '/data0/luohaibo/llama-weights/llama-2-7b/'
save_model_path = '/data0/luohaibo/llm/output'
peft_model_path = '/data0/luohaibo/llm/output/checkpoint-22000/'  # may be modified

base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
)
lora_model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
)

model = lora_model.merge_and_unload()

lora_model.train(False)

tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
model.save_pretrained(f"{save_model_path}-merged")
tokenizer.save_pretrained(f"{save_model_path}-merged")
