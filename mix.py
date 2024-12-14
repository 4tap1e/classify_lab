import pandas as pd
from openpyxl import load_workbook
from pre4data import CPC_change

#将timei.xlsx的sheet合并为ti_xxx.xlsx
def merge_sheets_by_rows(data, output_file, sheets):
    """按行整合sheet012的数据"""
    print(f"开始处理...")
    print(f"输出文件: {output_file}")
    
    # 读取各个sheet的数据
    dataframes = []
    
    for i, sheet in enumerate(sheets):
        try:
            print(f"\n正在读取 {sheet}...")
            df = data[sheet]
            # 添加来源标识列
            # df['sheet_source'] = sheet
            if i > 0:  # 只对第二个及以后的sheet做列名修改
                df.columns = [f"{col}_{i}" if col in data[sheets[0]].columns else col for col in df.columns]
            print(f"成功读取 {sheet}, 数据形状: {df.shape}")
            dataframes.append(df)
            
        except Exception as e:
            print(f"读取{sheet}时出错: {str(e)}")
            continue
    
    if not dataframes:
        raise Exception("没有成功读取任何sheet数据")
    
    print("\n正在合并数据...")
    # 按行连接数据框
    merged_data = pd.concat(dataframes, axis=1)
    
    # 显示最终数据的基本信息
    print(f"\n最终数据形状: {merged_data.shape}")
    print(f"列数量: {len(merged_data.columns)}")
    
    # 导出到Excel
    print(f"\n正在导出到: {output_file}")
    merged_data.to_excel(output_file, index=False)
    print("导出完成!")
    
    return merged_data

# def sheet_clean(inputdata, sheets):
#     data = inputdata
#     sheet_names = sheets

#     for sheet_name in sheet_names:
#         df = data[sheet_name]
#         cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
#         df = df.drop(columns=cols_to_drop)
#         print(f"Sheet '{sheet_name}' 删除了 {len(cols_to_drop)} 列")
#         data[sheet_name] = df
    
#     return data
#     # with pd.ExcelWriter(output_file) as writer:
#     #     for sheet_name, df in data.items():
#     #         df.to_excel(writer, sheet_name=sheet_name, index=False)

#     # print(f"已将每个sheet中列元素完全一样的列删除, 修改后的文件已保存为 {output_file}")

def main():
    data_path = 'D:/PycharmProject/classification/time1_1.xlsx'
    output_path = 'D:/PycharmProject/classification/t1_s123.xlsx'
    sheets = ['sheet1', 'sheet2', 'sheet3'] # , 'sheet1', 'sheet2', 'sheet3'

    data = pd.read_excel(data_path, sheet_name=None)
    # data = sheet_clean(data, sheets) 
    # cols = [col for col in data.columns if col != 'CPC'] + ['CPC']
    # data = data[cols]
    # data = CPC_change(data)
    data_out = merge_sheets_by_rows(data, output_path, sheets)
    print(data_out.shape)

if __name__ == "__main__":
    main()