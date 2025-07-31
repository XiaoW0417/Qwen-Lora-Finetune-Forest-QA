import pandas as pd
import json

# 1. 读取 Excel 文件（请替换为你的实际路径）
df = pd.read_excel("processed_data.xlsx")  # 文件名自己换成你实际的
df.dropna(subset=["prompt", "answer"], inplace=True)  # 去除空行
df["answer"] = df["answer"].str.replace("\n", " ")    # 去除回答中的换行符

# 2. 确保包含 'prompt' 和 'answer' 两列
if not all(col in df.columns for col in ["prompt", "answer"]):
    raise ValueError("Excel 文件必须包含 'prompt' 和 'answer' 两列")

# 3. 转为字典列表格式
qa_data = df[["prompt", "answer"]].to_dict(orient="records")

# 4. 保存为 JSON 文件
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(qa_data, f, ensure_ascii=False, indent=2)

print("✅ 成功将 Excel 数据保存为 JSON！")
