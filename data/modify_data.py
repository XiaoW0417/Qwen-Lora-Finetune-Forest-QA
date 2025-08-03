import json
import re

def remove_brackets(text):
    # 删除所有括号及括号内的内容，包括中文和英文括号
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r' ', '，', text)
    return text.strip()

def clean_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        item['prompt'] = remove_brackets(item['prompt'])
        item['answer'] = remove_brackets(item['answer'])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    input_file = 'train.json'   # 替换为你的原始文件路径
    output_file = 'train.json'  # 清洗后的输出路径
    clean_json_file(input_file, output_file)
