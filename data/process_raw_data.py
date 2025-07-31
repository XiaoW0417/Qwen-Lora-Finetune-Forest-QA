import pandas as pd


def delete_rows_with_empty_first_column(input_file, output_file):
    """
    删除Excel中第一列为空的行
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    # 读取Excel文件，header=None确保不自动识别列名[1,2](@ref)
    df = pd.read_excel(input_file, header=None)

    # 检查第一列（列索引0）是否为空，保留非空行[7](@ref)
    df_cleaned = df[df[0].notna()]

    # 保存结果，不保留索引和原有列名[2](@ref)
    df_cleaned.to_excel(output_file, index=False, header=False)
    print(f"处理完成，结果已保存至 {output_file}")


# 示例调用
delete_rows_with_empty_first_column("raw_data.xlsx", "processed_data.xlsx")