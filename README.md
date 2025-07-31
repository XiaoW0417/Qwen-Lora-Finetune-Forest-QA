# 🌲 Forest-QA: 微调 Qwen-1.5-1.8B 的林业问答系统

> 本项目基于 Qwen-1.5-1.8B 模型，利用 LoRA方法，在构造的小样本林业知识问答数据集上进行指令微调，最终部署为一个本地问答系统（Gradio Demo）。旨在验证小样本领域知识微调的可行性，并构建可复用、可扩展的 Forestry QA 基线系统。

---

## 📂 项目结构

```bash
Forest-QA/
├── data/
├── configs/                  
├── app/              
├── model/        
├── results/    
├── requirements.txt          
├── README.md                 

```

---

## 🧠 模型与方法

- **基座模型**：Qwen-1.5-1.8B (4bit量化加载)
- **微调方式**：LoRA (QLoRA 低资源微调)
- **训练框架**：Transformers + PEFT + bitsandbytes
- **适配模块**：（target_modules=["q_proj", "v_proj"],）

---

## 📊 数据集说明

训练数据构造自 400 条林业相关 QA 对话，包括部分人工标注、ChatGPT+DeepseekV3+豆包辅助生成生成结果。格式如下：

```json
[
  {
    "prompt": "林地面积变化的主要原因有哪些？",
    "answer": "林地面积变化的主要原因包括自然灾害、城市扩张、农业用途转换..."
  }
  
]
```

> ⚠ 本项目暂未使用验证集，后续计划增加交叉验证或K折分割增强泛化能力。

---

## ⚙️ 训练方式

```bash
python scripts/train_lora.py \
  --base_model "Qwen/Qwen-1_5-1.8B" \
  --data_path "./data/train.json" \
  --output_dir "./results" \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_epochs 5 \
  --lr 1e-4 \
  --lora_config "./config/lora_config.json"
```

> 训练过程通过 TensorBoard 实时可视化，支持保存中间checkpoint与最终LoRA Adapter。

---

Gradio UI 部署：

```bash
cd app
python app.py
```

界面功能：
- 输入自定义问题
- 实时返回模型答案
- 支持多轮清空/记录日志

---

## 🧩 lora_config.json 示例

```json
{
  "r":8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.1,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

---

## 📦 环境依赖

requirements.txt

---


## 🧑‍💻 作者信息

- 👨‍🎓 作者：WangJun
- 📬 联系方式：wangjun1704@bjfu.edu.cn

---

> 若您觉得本项目有帮助，欢迎 ⭐ ！

