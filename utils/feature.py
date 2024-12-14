import pandas as pd
import os
from collections import defaultdict

def print_sheet_features(file_path):
    """打印每个sheet的特征列名"""
    print(f"正在分析文件: {file_path}")
    print(f"文件是否存在: {os.path.exists(file_path)}\n")
    
    # 读取各个sheet的数据
    sheets = ['sheet0', 'sheet1', 'sheet2']
    
    for sheet in sheets:
        try:
            print(f"{'='*20} {sheet} {'='*20}")
            # 读取sheet
            df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')
            
            # 打印基本信息
            print(f"\n数据形状: {df.shape}")
            print(f"列数量: {len(df.columns)}")
            
            # 查找CPC列
            cpc_cols = [col for col in df.columns if 'CPC' in str(col).upper()]
            print(f"\nCPC相关列 ({len(cpc_cols)}):")
            for col in cpc_cols:
                print(f"- {col}")
            
            # 按类别组织并打印其他特征列
            other_cols = [col for col in df.columns if 'CPC' not in str(col).upper()]
            
            # 按前缀分组
            col_groups = {}
            for col in other_cols:
                prefix = col.split('_')[0] if '_' in col else 'other'
                if prefix not in col_groups:
                    col_groups[prefix] = []
                col_groups[prefix].append(col)
            
            # 打印分组结果
            print(f"\n其他特征列 (按类别分组):")
            for prefix, cols in col_groups.items():
                print(f"\n{prefix} ({len(cols)}):")
                for col in sorted(cols):
                    print(f"- {col}")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"读取{sheet}时出错: {str(e)}\n")


def analyze_sheets_features(file_path):
    """分析每个sheet的特征列并找出共同特征"""
    print(f"正在分析文件: {file_path}")
    print(f"文件是否存在: {os.path.exists(file_path)}\n")
    
    # 读取各个sheet的数据
    sheets = ['sheet0', 'sheet1', 'sheet2']
    sheet_columns = {}  # 存储每个sheet的列名
    all_columns = defaultdict(list)  # 按前缀分组存储所有列名
    
    # 读取所有sheet的列名
    for sheet in sheets:
        try:
            print(f"{'='*20} {sheet} {'='*20}")
            df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')
            
            # 存储基本信息
            sheet_columns[sheet] = set(df.columns)
            print(f"数据形状: {df.shape}")
            print(f"列数量: {len(df.columns)}")
            
            # 按类别组织列名
            for col in df.columns:
                prefix = col.split('_')[0] if '_' in col else 'other'
                all_columns[prefix].append((col, sheet))
            
            print("\n")
            
        except Exception as e:
            print(f"读取{sheet}时出错: {str(e)}\n")
    
    # 找出共同特征
    common_features = set.intersection(*sheet_columns.values())
    print(f"{'='*20} 共同特征分析 {'='*20}")
    print(f"\n在所有sheet中共同存在的特征数量: {len(common_features)}")
    
    # 按前缀组织共同特征
    common_by_prefix = defaultdict(list)
    for col in common_features:
        prefix = col.split('_')[0] if '_' in col else 'other'
        common_by_prefix[prefix].append(col)
    
    print("\n共同特征列表 (按类别分组):")
    for prefix, cols in sorted(common_by_prefix.items()):
        print(f"\n{prefix} ({len(cols)}):")
        for col in sorted(cols):
            print(f"- {col}")
    
    # 分析每个sheet的独特特征
    print(f"\n{'='*20} 独特特征分析 {'='*20}")
    for sheet in sheets:
        unique_features = sheet_columns[sheet] - set.union(*(
            sheet_columns[other_sheet] for other_sheet in sheets if other_sheet != sheet
        ))
        print(f"\n{sheet} 独有的特征数量: {len(unique_features)}")
        if unique_features:
            print("前10个独有特征示例:")
            for col in sorted(list(unique_features))[:10]:
                print(f"- {col}")
    
    # 分析sheet之间的特征重叠情况
    print(f"\n{'='*20} 特征重叠分析 {'='*20}")
    for i, sheet1 in enumerate(sheets):
        for sheet2 in sheets[i+1:]:
            common = sheet_columns[sheet1] & sheet_columns[sheet2]
            print(f"\n{sheet1} 和 {sheet2} 之间共同的特征数量: {len(common)}")

#
def main():
    # 文件路径
    file_path = 'D:/PycharmProject/classification/time0.xlsx'  # 请根据实际情况修改路径
    
    try:
        analyze_sheets_features(file_path)
        
        # 将结果保存到文本文件
        output_file = 'feature_columns.txt'
        print(f"\n正在将结果保存到文件: {output_file}")
        
        # 重定向标准输出到文件
        import sys
        original_stdout = sys.stdout
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            analyze_sheets_features(file_path)
            sys.stdout = original_stdout
        print(f"结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()