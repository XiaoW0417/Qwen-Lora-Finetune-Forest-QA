
import pandas as pd

def delete_rows_with_empty_first_column(input_file, output_file):
    """
    删除Excel中第一列为空的行，
    并清除第二列数据中以中文逗号‘，’开头的字符
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    # 读取Excel文件
    df = pd.read_excel(input_file, header=None)

    # 删除第一列为空的行
    df_cleaned = df[df[0].notna()]

    # 安全替换：删除第二列中以中文逗号开头的字符
    df_cleaned.loc[:, 1] = df_cleaned[1].astype(str).str.lstrip('，')

    # 保存处理结果
    df_cleaned.to_excel(output_file, index=False, header=False)
    print(f"处理完成，结果已保存至 {output_file}")


# 示例调用
delete_rows_with_empty_first_column("raw_data.xlsx", "processed_data.xlsx")
