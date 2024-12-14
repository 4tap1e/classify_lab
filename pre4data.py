import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV
from keras._tf_keras.keras.layers import Input, Dense
from keras._tf_keras.keras.models import Model
from keras import regularizers
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def lasso_dimension_reduction(data, cv=5, n_alphas=100, alpha_range=(-5,-2)):
    """
    使用LASSO进行特征降维
    
    参数:
    data: pd.DataFrame, 输入数据
    cv: int, 交叉验证折数, 默认5
    n_alphas: int, alpha参数的数量, 默认100 
    alpha_range: tuple, alpha参数的范围(start,end)，默认(-10,-1)
    返回:
    data_lasso: pd.DataFrame, 降维后的数据, 包含label列和选中的特征
    selected_features: list, 被选中的特征名称列表
    best_alpha: float, 最佳的alpha值
    """
    # 数据预处理
    X = data[data.columns[:-1]]
    y = data['label']
    
    # 标准化
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=data.columns[:-1])
    
    # 设置alpha参数范围
    alphas = np.logspace(alpha_range[0], alpha_range[1], n_alphas)
    
    # 使用LassoCV进行特征选择
    model_lassoCV = LassoCV(alphas=alphas, cv=cv, max_iter=10000, random_state=42, tol=1e-2, selection='random').fit(X, y)
    
    # 获取特征系数
    coef = pd.Series(model_lassoCV.coef_, index=X.columns)
    # 创建特征-系数数据框并排序
    coef_df = pd.DataFrame({
        'feature': coef.index,
        'coefficient': coef.values
    })
    coef_df['abs_coef'] = abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    # 打印所有非零系数
    print("\n特征系数表 (按绝对值排序):")
    print("非零系数数量:", sum(coef_df['coefficient'] != 0))
    print(coef_df[coef_df['coefficient'] != 0].to_string())
    
    # 获取非零系数的特征
    selected_features = coef_df[coef_df['coefficient'] != 0]['feature'].tolist()
    
    # 构建降维后的数据集
    data_lasso = pd.DataFrame()
    for feature in selected_features:
        data_lasso[feature] = data[feature]
    data_lasso['label'] = data['label']
    return data_lasso, selected_features

def extract(data1, data2, n_features=20):
    """
    从两个数据集中提取重要特征
    
    参数:
        data1: pd.DataFrame, 包含特征重要性排序的数据框
        data2: pd.DataFrame, 包含完整特征和标签的数据框
        n_features: int, 需要提取的特征数量, 默认为20
    返回:
        pd.DataFrame: 提取出的重要特征数据框
    """
    try:
        # 获取前n个重要特征的列名
        selected_columns = data1.iloc[:n_features, 0].tolist()
        selected_columns = [str(col).strip() for col in selected_columns]
        
        # 获取标签
        label = data2.iloc[:, -1]
        
        # 验证特征列是否存在
        valid_columns = [col for col in selected_columns if col in data2.columns]
        if not valid_columns:
            raise Exception("没有找到匹配的列名")
            
        # 提取重要特征
        data_important = data2[selected_columns]
        data_important['label'] = label
        
        return data_important
        
    except Exception as e:
        print(f"特征提取过程发生错误: {str(e)}")
        return None

def get_importantfeature_xgb(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 初始化 XGBoost 分类器
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    feature_selector = SelectFromModel(estimator=xgb_clf, max_features=26, prefit=False)

    # ... 模型训练和评估 ...
    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('classifier', xgb_clf)
    ])
    
    param_grid = { 
        'feature_selection__estimator__n_estimators': [128, 180],     #树的数量 
        'classifier__max_depth': [4, 6, 8],                           #树最大深度
        'classifier__learning_rate': [0.01, 0.06, 0.1],                 
        'classifier__subsample': [0.78, 0.86, 0.93],                     #样本采样比例
        'classifier__colsample_bytree': [0.78, 0.85, 0.9]               #特征采样比例
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("开始训练模型...")
    grid_search.fit(X_train, y_train)
    print("最佳参数组合：", grid_search.best_params_)
    print("最佳 ROC-AUC 分数：", grid_search.best_score_)
    best_model = grid_search.best_estimator_

    # ... 计算特征重要性 ...
    selected_features = best_model.named_steps['feature_selection'].get_support(indices=True)
    selected_features_names = X.columns[selected_features]

    importances = best_model.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        '特征': selected_features_names,
        '重要性': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='重要性', ascending=False)
    print(feature_importance_df.shape)
    selected_data = X[selected_features_names].copy()
    selected_data['label'] = y
    return selected_data, selected_features_names

def CPC_change(data):
    """
    将CPC列转换为二分类标签
    CPC值为1,2的设为0
    CPC值为3,4,5的设为1
    """
    try:
        df = data       
        if 'CPC' not in df.columns:
            raise ValueError("未找到CPC列")
        
        df['label'] = df['CPC'].apply(lambda x: 0 if x <= 2 else 1)
        df = df.drop('CPC', axis=1)
        
        value_counts = df['label'].value_counts()
        print("\n标签分布:")
        print(f"标签0[CPC = 1,2]的样本数: {value_counts.get(0, 0)}")
        print(f"标签1[CPC = 3,4,5]的样本数: {value_counts.get(1, 0)}")
        print(f"标签1比例为: {value_counts.get(1, 0) / (value_counts.get(0, 0) + value_counts.get(1, 0))}")
        print(f"标签0比例为: {value_counts.get(0, 0) / (value_counts.get(0, 0) + value_counts.get(1, 0))}")
        
        return df
        
    except Exception as e:
        print(f"CPC转换时出错: {str(e)}")
        return None

def drop_columns(data, dropdata):
    try:
        existing_columns = [col for col in dropdata if col in data.columns]
        if not existing_columns:
            print("没有找到任何需要删除的列。")
            return data

        # 删除这些列
        print(f"\n原始列数: {len(data.columns)}")
        df = data.drop(existing_columns, axis=1)
        print(f"删除后的列数: {len(df.columns)}")
        return df
        
    except Exception as e:
        print(f"删除指定数据列时出错: {str(e)}")
        return None

def analyze_columns(data, var_threshold=3):
    """分析列数据的相似性"""
    try:
        df = data.drop('label', axis=1) 
        print(f"\n{'='*30} 列数据分析 {'='*30}")
        
        # 存储列内值完全相同的列名和对应的值
        same_cols = {}  # 格式: {列名: 该列的唯一值}
        variance_cols = [] 
        
        for col in df.columns:
            unique_values = df[col].unique()
            # 如果该列只有一个唯一值
            if len(unique_values) == 1:
                same_cols[col] = unique_values[0]
        
        # 输出结果
        print(f"\n总列数: {len(df.columns)}")
        print(f"列内值完全相同的列数: {len(same_cols)}")
        print("\n 列内值完全相同的列:")

        print(f"\n{'='*30} 方差分析 {'='*30}")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # 检查是否所有值都相同（如12,12,12这种情况）
                if df[col].nunique() == 1:
                    continue
                    
                # 计算方差
                variance = df[col].var()
                if variance <= var_threshold:  
                    variance_cols.append(col)
        print(f"\n方差<{var_threshold}的列数: {len(variance_cols)}")
        return same_cols, variance_cols

    except Exception as e:
        print(f"预处理文件时出错: {str(e)}")
        return {}

def if_same(data1, data2):
    cols_1 = set(data1.columns)
    cols_2 = set(data2.columns)

    common_cols = cols_1.intersection(cols_2)
    diff_cols = cols_1.symmetric_difference(cols_2)
    print(f"相同列数: {len(common_cols)}")
    print(f"不同列数: {len(diff_cols)}")
    return common_cols == cols_1 == cols_2

def split_dataset(data, train_size=0.85, random_state=42):
    try:
        df = data
        print(f"数据形状: {df.shape}")
        X = df.iloc[:, :-1]  
        y = df.iloc[:, -1]   
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            train_size=train_size,
            random_state=random_state,
            stratify=y  # 保持标签分布一致
        )
        print("\n数据集划分结果:")
        print(f"训练集形状: {X_train.shape}")
        print(f"验证集形状: {X_val.shape}")
        print("\n标签分布:")
        print("训练集:")
        print(y_train.value_counts(normalize=True))
        print("\n验证集:")
        print(y_val.value_counts(normalize=True))
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        print(f"数据集划分时出错: {str(e)}")
        return None, None, None, None

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
    data = 'D:/PycharmProject/classification/t0_s012.xlsx'
    output_file = 'D:/PycharmProject/classification/t0_n1_2.xlsx'
    # 废弃特征
    dropdata = ['diagnostics_Image-original_Hash', 'diagnostics_Mask-original_Hash',
                'diagnostics_Image-original_Spacing','diagnostics_Image-original_Size',
                'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_Size',
                'diagnostics_Mask-original_BoundingBox','diagnostics_Mask-original_CenterOfMassIndex',
                'diagnostics_Mask-original_CenterOfMass','sheet_source']
    
    try:
        df = pd.read_excel(data, engine='openpyxl')
        same_cols, variance_cols = analyze_columns(df)
        df = drop_columns(df, dropdata, output_file)
        df = df.drop(columns=list(same_cols.keys()))
        # df = df.drop(columns=variance_cols)
        # df = CPC_change(df, output_file)
        df.to_excel(output_file, index=False)
        print(f"t0_n1_1.xlsx的形状: {df.shape}")      
    except Exception as e:
        print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    main()