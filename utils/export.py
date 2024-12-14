import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
from pre4data import analyze_columns

def analyze_output_file(data):
    """分析输出文件的列名和形状"""
    print(f"\n{'='*30} 输出文件分析 {'='*30}")
    
    try:
        df = data
        print(f"\n基本信息:")
        print(f"数据形状: {df.shape}")
        print(f"总列数: {len(df.columns)}")
        
        if 'label' in df.columns:
            print("\n标签分布:")
            print(df['label'].value_counts())
        
    except Exception as e:
        print(f"分析输出文件时出错: {str(e)}")

def merge_sheets_by_rows(data, output_file):
    """按行整合sheet012的数据"""
    print(f"开始处理...")
    print(f"输出文件: {output_file}")
    
    # 读取各个sheet的数据
    sheets = ['sheet0', 'sheet1', 'sheet2']
    dataframes = []
    
    for sheet in sheets:
        try:
            print(f"\n正在读取 {sheet}...")
            df = data[sheet]
            # 添加来源标识列
            # df['sheet_source'] = sheet
            print(f"成功读取 {sheet}, 数据形状: {df.shape}")
            dataframes.append(df)
            
        except Exception as e:
            print(f"读取{sheet}时出错: {str(e)}")
            continue
    
    if not dataframes:
        raise Exception("没有成功读取任何sheet数据")
    
    print("\n正在合并数据...")
    # 按行连接数据框
    merged_data = pd.concat(dataframes, axis=0, ignore_index=True)
    
    # 显示最终数据的基本信息
    print(f"\n最终数据形状: {merged_data.shape}")
    print(f"列数量: {len(merged_data.columns)}")
    
    # 导出到Excel
    print(f"\n正在导出到: {output_file}")
    merged_data.to_excel(output_file, index=False)
    print("导出完成!")
    
    return merged_data

#将time0.xlsx的sheet012合并为t0_s012.xlsx
def main():
    # 设置文件路径
    input_file = 'D:/PycharmProject/classification/time0.xlsx'
    output_file = 'D:/PycharmProject/classification/t0_s012_1.xlsx'
    
    try:
        # 执行合并操作
        df = pd.read_excel(input_file, engine='openpyxl')
        sheets = ['sheet0', 'sheet1', 'sheet2']
        data = pd.DataFrame()
        variance_cols_by_sheet = {}

        for sheet in sheets:
            try:
                print(f"\n正在读取 {sheet}...")
                df = pd.read_excel(input_file, sheet_name=sheet, engine='openpyxl')
                same_cols, variance_cols = analyze_columns(df)
                # df_filtered = df.drop(columns=list(same_cols.keys()))
                # df = df_filtered.drop(columns=variance_cols)
                print(f"成功读取 {sheet}, 数据形状: {df.shape}")
                variance_cols_by_sheet[sheet] = set(variance_cols)
                # data = pd.concat([data, df], axis=0, ignore_index=True)
            except Exception as e:
                print(f"读取{sheet}时出错: {str(e)}")
         
        # merged_data = merge_sheets_by_rows(data, output_file)
        common_variance_cols = set.intersection(*variance_cols_by_sheet.values())
        print(f"\n在所有sheet中方差都小于1的列数: {len(common_variance_cols)}")
        
        for sheet in sheets:
            try:
                df = pd.read_excel(input_file, sheet_name=sheet, engine='openpyxl')
                same_cols, _ = analyze_columns(df)
                # 只删除相同值的列和在所有sheet中都方差小于1的列
                df_filtered = df.drop(columns=list(same_cols.keys()))
                df_filtered = df_filtered.drop(columns=list(common_variance_cols))
                print(f"成功读取 {sheet}, 数据形状: {df_filtered.shape}")
                data = pd.concat([data, df_filtered], axis=0, ignore_index=True)
            except Exception as e:
                print(f"处理{sheet}时出错: {str(e)}")
        data.to_excel(output_file, index=False)
        print(f"t0_s012_1.xlsx的形状: {data.shape}")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    main()