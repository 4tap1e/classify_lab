import os 

def rename_files_in_subfolders(results_folder):
    """
    将 results 文件夹下所有子文件夹里的文件命名为子文件夹的名字。
    如果有多个文件，则命名为 子文件夹名_1, 子文件夹名_2。
    如果只有一个文件，则直接命名为子文件夹名。

    :param results_folder: str, 主文件夹路径
    """
    # 遍历主文件夹下的所有子文件夹
    for subfolder in os.listdir(results_folder):
        subfolder_path = os.path.join(results_folder, subfolder)
        
        # 确认是子文件夹
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            num_files = len(files)
            
            # 遍历子文件夹中的文件
            for index, file in enumerate(files):
                file_path = os.path.join(subfolder_path, file)
                
                # 确保是文件而不是目录
                if os.path.isfile(file_path):
                    # 构造新的文件名
                    if num_files == 1:
                        new_name = f"{subfolder}{os.path.splitext(file)[-1]}"  # 子文件夹名+原文件扩展名
                    else:
                        new_name = f"{subfolder}_{index+1}{os.path.splitext(file)[-1]}"  # 子文件夹名_索引+扩展名
                    
                    new_path = os.path.join(subfolder_path, new_name)
                    
                    # 重命名文件
                    os.rename(file_path, new_path)
                    print(f"重命名文件: {file_path} -> {new_path}")
    
    print("重命名完成！")

results_folder = "SelfResultsssss/t1_s0"  
rename_files_in_subfolders(results_folder)