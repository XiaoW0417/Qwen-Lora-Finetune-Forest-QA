from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import torch
# import evaluate

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B", device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "../results/checkpoints/checkpoint-750")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B")

# Load test set
with open('../data/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

preds, refs = [], []

for sample in test_data:
    prompt = sample['prompt'] + '\n答：'
    reference = sample['answer']

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    # print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
    output = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    preds.append(output)
    refs.append(reference)

# rouge = evaluate.load("rouge")

from rouge import Rouge
rouge = Rouge()

preds = [" ".join(pred.replace(" ", "")) for pred in preds]
refs = [" ".join(label.replace(" ", "")) for label in refs]

print(f'preds:{preds}\nrefs:{refs}')


results = rouge.get_scores(preds, refs)

print(results)
