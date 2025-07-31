import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# -------- 模型加载 --------
def load_model(
        model_name="Qwen/Qwen1.5-1.8B",
        adapter_path="../results/adapter"
):
    # 4bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载LoRA Adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer


# 初始化模型与tokenizer
model, tokenizer = load_model()


# -------- 推理函数 --------
def answer_question(question):
    prompt = f"用户：{question}\n助手："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("助手：")[-1].strip()


# -------- Gradio UI 构建 --------
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        lines=3,
        placeholder="请输入你的问题，例如：如何保护森林健康？",
        label="提问框"
    ),
    outputs=gr.Textbox(
        lines=5,
        label="模型回答"
    ),
    title="🔍 Forest--微调问答机器人 (LoRA + Qwen)",
    description="本系统基于 Qwen-1.5 1.8B 模型，结合 LoRA 微调构建，支持本地轻量化部署问答。",
    theme="default"
)

if __name__ == "__main__":
    demo.launch(show_api=False)
