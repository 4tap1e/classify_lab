import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(metrics_history, result_folder):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8))
    
    # 绘制每个指标的变化趋势
    for metric_name, values in metrics_history.items():
        plt.plot(range(1, len(values) + 1), values, marker='o', label=metric_name)
    
    plt.xlabel('轮次')
    plt.ylabel('指标值')
    plt.title('模型性能指标变化趋势')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(result_folder, 'metrics_trend.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"性能指标趋势图已保存到: {plot_path}")

def main():
    try:#提取重要特征
        data_path = 'D:/PycharmProject/classification/t0_n1_2.xlsx'
        output_path = 'D:/PycharmProject/classification/t0_train_20.xlsx'
        feature_path = 'D:/PycharmProject/classification/t0_train0.xlsx'

        train_data = pd.read_excel(feature_path, engine='openpyxl')
        print(f"成功读取文件: {feature_path}")
        train_data = train_data.head(20)
        selected_columns = train_data.iloc[:, 0].tolist()
        selected_columns = [str(col).strip() for col in selected_columns]

        data = pd.read_excel(data_path, engine='openpyxl')
        label = data.iloc[:, -1]
        print(f"成功读取label")

        valid_columns = [col for col in selected_columns if col in data.columns]
        if not valid_columns:
            raise Exception("没有找到匹配的列名")

        data_important = data[selected_columns]
        data_important['label'] = label
        data_important.to_excel(output_path, index=False)
        print(f"已保存重要特征到: {output_path}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()