import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import os
from datetime import datetime
from pre4data import plot_metrics, lasso_dimension_reduction, get_importantfeature_xgb, if_same
import xgboost as xgb
from sklearn.svm import SVC

def get_next_result_folder():
    base_path = 'D:/PycharmProject/classification/results'
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return os.path.join(base_path, 'results_1')
    
    # 查找现有的results_i文件夹
    existing_folders = [d for d in os.listdir(base_path) 
                       if os.path.isdir(os.path.join(base_path, d)) 
                       and d.startswith('results_')]
    
    if not existing_folders:
        return os.path.join(base_path, 'results_1')
    
    # 获取现有文件夹的最大编号
    max_num = max([int(f.split('_')[1]) for f in existing_folders])
    
    # 返回下一个编号的文件夹路径
    return os.path.join(base_path, f'results_{max_num + 1}')

def save_results(results_text, result_folder):
    """保存结果到results.txt文件"""
    os.makedirs(result_folder, exist_ok=True)
    
    result_file = os.path.join(result_folder, 'results.txt')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_with_timestamp = f"实验时间: {timestamp}\n\n{results_text}"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(results_with_timestamp)
    
    print(f"\n结果已保存到: {result_file}")

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    ACC = (tp + tn) / (tp + tn + fp + fn)
    Recall = tp / (tp + fn) if (tp + fn) != 0 else 0  
    Specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  
    Precision = tp / (tp + fp) if (tp + fp) != 0 else 0  
    NPV = tn / (tn + fn) if (tn + fn) != 0 else 0  
    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    
    return ACC, Recall, Specificity, Precision, NPV, roc_auc

def main():
    data_train_path = 'D:/PycharmProject/classification/t0_s0_train.xlsx' # E:/CTSeg/code-zhc/time0.xlsx  D:/PycharmProject/classification/time0.xlsx
    data_train = pd.read_excel(data_train_path)

    print(f"训练集形状: {data_train.shape}")
    result_folder = get_next_result_folder()

    X_index = data_train.iloc[:, :-1]
    y_index = data_train.iloc[:, -1]
    scaler = StandardScaler()
    print(f"数据划分完毕")

    # 初始化分类器
    # clf = GaussianNB()  #priors=[0.5, 0.5]
    scale_pos_weight = len(y_index[y_index == 0]) / len(y_index[y_index == 1])
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': ['logloss'],
        'learning_rate': 0.012,
        'max_depth': 6,
        'n_estimators': 300,
        'subsample': 0.73,         # 用于构建每棵树的样本比例
        'colsample_bytree': 0.74,  # 控制每棵树在构建时使用的特征比例
        # 'scale_pos_weight': 1.4,   # 根据实际正负样本比例设置权重 len(y[y==0]) / len(y[y==1]),
        'gamma': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'min_child_weight': 1,     # 降低以增加模型灵活性
        'random_state': 42,
        # 'tree_method': 'hist',  
        # 'device': 'cuda',
    }
    clf = xgb.XGBClassifier(**xgb_params)
    # clf = LogisticRegression()
    # clf = SVC(
    #     kernel='rbf',              # 使用RBF核函数
    #     C=0.95,                     # 正则化参数
    #     gamma='auto',              # scale
    #     probability=True,          # 启用概率估计
    #     class_weight='balanced', # 处理类别不平衡 
    #     random_state=42
    # )
    random_var = [42, 46, 52]



   
    # # 参数网格
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'max_depth': [5, 6, 7],
    #     'n_estimators': [100, 200, 500],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.8, 0.9, 1.0],
    # }

    # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    # grid_search.fit(X_train_scaled, y_train)


    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

    acc_scores = []
    recall_scores = []
    specificity_scores = []
    precision_scores = []
    npv_scores = []
    auc_scores = []

    metrics_history = {
        'ACC': [], 'Recall': [], 'Specificity': [], 'Precision': [], 'NPV': [], 'AUC': [], 
    }
    
    for fold, (train_index, test_index) in enumerate(kf.split(X_index, y_index), 1):
        print(f"==============第 {fold} 折交叉验证==============")
        print(f"data shape is : {data_train.shape}")
        # 特征降维
        data_train_1 = data_train.iloc[train_index]
        print(f"data for {fold} 折 train shape is : {data_train_1.shape}")
        data_train_lasso, selected_features = lasso_dimension_reduction(data_train_1)
        # data_train_lasso, selected_features = get_importantfeature_xgb(data_train)
        X_train = data_train_lasso.iloc[:, :-1]
        y_train = data_train_lasso.iloc[:, -1]
        data_val = data_train.iloc[test_index]
        X_val = data_val.iloc[:, :-1]
        y_val = data_val.iloc[:, -1]
        X_val = X_val[selected_features]
        t = if_same(X_train, X_val)
        if t:
            print(f"X_train and X_val is same")
        else:
            print(f"X_train is not same as X_val")
        # y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        print(f"第 {fold} 折交叉验证X_test.shape: {X_val.shape}, X_train.shape: {X_train.shape}")

        # 标准化
        X_train_scaled = scaler.fit_transform(X_train)  
        X_val_scaled = scaler.transform(X_val)

        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 0] = len(y_train) / (2 * (y_train == 0).sum())
        sample_weights[y_train == 1] = len(y_train) / (2 * (y_train == 1).sum())

        print(f"训练开始")
        # clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)         #贝叶斯分类器 , logic分类器
        clf.fit(X_train_scaled, y_train)   #SVM, xgb分类器
        print(f"训练完成")

        y_pred = clf.predict(X_val_scaled)
        y_prob = clf.predict_proba(X_val_scaled)[:, -1]

        ACC, Recall, Specificity, Precision, NPV, roc_auc = calculate_metrics(y_val, y_pred, y_prob)

        acc_scores.append(ACC)
        recall_scores.append(Recall)
        specificity_scores.append(Specificity)
        precision_scores.append(Precision)
        npv_scores.append(NPV)
        auc_scores.append(roc_auc)

        print(f"第 {fold} 折交叉验证ACC:{ACC:.3f} ± {np.std(acc_scores):.3f}, {Recall:.3f} ± {np.std(recall_scores):.3f},"
              f"{Specificity:.3f} ± {np.std(specificity_scores):.3f}, {Precision:.3f} ± {np.std(precision_scores):.3f},"    
              f"{NPV:.3f} ± {np.std(npv_scores):.3f}, {roc_auc:.3f} ± {np.std(auc_scores):.3f}")

    final_ACC = np.mean(acc_scores)
    final_Recall = np.mean(recall_scores)
    final_Specificity = np.mean(specificity_scores)
    final_Precision = np.mean(precision_scores)
    final_NPV = np.mean(npv_scores)
    final_AUC = np.mean(auc_scores)
    print("\n最终测试集的具体指标值:")
    print(f"准确率 (ACC): {final_ACC:.3f}")
    print(f"召回率 (Recall): {final_Recall:.3f}")
    print(f"特异性 (Specificity): {final_Specificity:.3f}")
    print(f"精确率 (PPV): {final_Precision:.3f}")
    print(f"阴性预测值 (NPV): {final_NPV:.3f}")
    print(f"AUC值: {final_AUC:.3f}")

    final_results = {
        'ACC': f"{final_ACC:.3f}",
        'Recall': f"{final_Recall:.3f}",
        'Specificity': f"{final_Specificity:.3f}",
        'PPV': f"{final_Precision:.3f}",
        'NPV': f"{final_NPV:.3f}",
        'AUC': f"{final_AUC:.3f}"
    }
    results = "\n"
    for metric, value in final_results.items():
        results += f"{metric}: {value}\n"
    save_results(results, result_folder)

if __name__ == "__main__":
    main()

