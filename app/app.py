import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel
import threading


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
device = next(model.parameters()).device


# -------- 流式推理函数 --------
def generate_answer_stream(question, max_new_tokens=128, temperature=0.75, top_p=0.9, repetition_penalty=1.2):
    prompt = f"用户：{question}\n助手："
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    partial_response = ""
    for token in streamer:
        partial_response += token
        # 去除多余空白和多余连字符，连续空格合并为一个空格
        cleaned = partial_response.replace("-", "").replace("\n ", "\n").strip()
        # 合并多余空格
        cleaned = ' '.join(cleaned.split())
        yield cleaned


# -------- Gradio UI --------
with gr.Blocks(css="""
    body { background-color: #f7f7f8; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .chatbot-message.user div {
        background-color: #0b93f6; color: white;
        border-radius: 18px 18px 0 18px; padding: 10px 15px; margin: 5px 10px 5px 50px; max-width: 80%;
        white-space: pre-wrap; word-wrap: break-word;
    }
    .chatbot-message.bot div {
        background-color: #e5e5ea; color: black;
        border-radius: 18px 18px 18px 0; padding: 10px 15px; margin: 5px 50px 5px 10px; max-width: 80%;
        white-space: pre-wrap; word-wrap: break-word;
    }
    #chatbot {
        height: 500px; overflow-y: auto; border-radius: 12px; background-color: white;
        padding: 10px; box-shadow: 0 2px 5px rgb(0 0 0 / 0.1);
    }
    #input_text textarea {
        border-radius: 18px; padding: 12px 20px; font-size: 16px; border: 1px solid #ccc;
        box-shadow: none; width: 100%; resize: none; background-color: white;
    }
    button {
        border-radius: 12px; background-color: #0b93f6; color: white; border: none;
        padding: 10px 20px; margin-left: 10px; cursor: pointer; font-weight: 600;
    }
    button:hover { background-color: #0a83e6; }
""") as demo:

    gr.Markdown("## 🔥 Forest 微调问答机器人 (LoRA + Qwen)")

    chatbot = gr.Chatbot(elem_id="chatbot", type="messages")
    msg = gr.Textbox(placeholder="请输入问题，例如：如何保护森林健康？", lines=2, elem_id="input_text")
    submit_btn = gr.Button("提交")
    clear_btn = gr.Button("清除聊天")

    history = []

    def user_submit(user_message):
        history.append({"role": "user", "content": user_message})
        return "", history

    def bot_reply(history):
        question = history[-1]["content"]
        partial = ""
        for chunk in generate_answer_stream(question):
            partial = chunk
            # 用 assistant 角色更新最后一条回复
            if history and history[-1].get("role") == "assistant":
                history[-1]["content"] = partial
            else:
                history.append({"role": "assistant", "content": partial})
            yield history

    msg.submit(user_submit, inputs=msg, outputs=[msg, chatbot], queue=False).then(
        bot_reply, inputs=chatbot, outputs=chatbot
    )
    submit_btn.click(user_submit, inputs=msg, outputs=[msg, chatbot], queue=False).then(
        bot_reply, inputs=chatbot, outputs=chatbot
    )
    clear_btn.click(lambda: [], None, chatbot, queue=False)
if __name__ == "__main__":
    demo.launch()
