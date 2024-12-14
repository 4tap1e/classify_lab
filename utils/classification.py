import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import os
from datetime import datetime
from pre4data import plot_metrics
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
    data_train_path = 'D:/PycharmProject/classification/t1_s3_train_xgb.xlsx' # E:/CTSeg/code-zhc/time0.xlsx  D:/PycharmProject/classification/time0.xlsx
    data_test_path = 'D:/PycharmProject/classification/t1_s3_test_xgb.xlsx'

    try:
        data_train = pd.read_excel(data_train_path)
        data_test = pd.read_excel(data_test_path)
        print(f"训练集形状: {data_train.shape}, 测试集形状: {data_test.shape}")
        result_folder = get_next_result_folder()

        X = data_train.iloc[:, :-1]
        X_test_final = data_test.iloc[:, :-1]
        y = data_train.iloc[:, -1]
        y_test_final = data_test.iloc[:, -1]

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test_final)

        best_model = None
        best_auc = 0

        # 初始化分类器
        clf = GaussianNB(priors=[0.5, 0.5])  #priors=[0.5, 0.5]

        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': ['logloss'],
            'learning_rate': 0.004,
            'max_depth': 4,
            'n_estimators': 400,
            'subsample': 0.78,         # 用于构建每棵树的样本比例
            'colsample_bytree': 0.78,  # 控制每棵树在构建时使用的特征比例
            'scale_pos_weight': 1.4,   # 根据实际正负样本比例设置权重 len(y[y==0]) / len(y[y==1]),
            'gamma': 0.1,
            'min_child_weight': 2,     # 降低以增加模型灵活性
            'random_state': 42
        }
        # clf = xgb.XGBClassifier(**xgb_params)
        n_samples = len(y)
        n_class_0 = len(y[y==0])
        n_class_1 = len(y[y==1])
        weight_0 = n_samples / (2 * n_class_0)
        weight_1 = n_samples / (2 * n_class_1)
        # clf = SVC(
        #     kernel='rbf',              # 使用RBF核函数
        #     C=0.1,                     # 正则化参数
        #     gamma='auto',              # scale
        #     probability=True,          # 启用概率估计
        #     # class_weight='balanced', # 处理类别不平衡
        #     class_weight={0: weight_0, 1: weight_1},
        #     random_state=42
        # )

        all_rounds_acc = []
        all_rounds_recall = []
        all_rounds_specificity = []
        all_rounds_precision = []
        all_rounds_npv = []
        all_rounds_auc = []
        random_var = [42, 46, 52]

        for round_num in range(3):
            print(f"\n第 {round_num + 1} 轮交叉验证：")
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_var[round_num])

            acc_scores = []
            recall_scores = []
            specificity_scores = []
            precision_scores = []
            npv_scores = []
            auc_scores = []
            metrics_history = {
                'ACC': [], 'Recall': [], 'Specificity': [], 'Precision': [], 'NPV': [], 'AUC': []
            }

            for fold, (train_index, test_index) in enumerate(kf.split(X_scaled, y), 1):
                X_train, X_val = X_scaled[train_index], X_scaled[test_index]
                y_train, y_val = y.iloc[train_index], y.iloc[test_index]

                sample_weights = np.ones(len(y_train))
                sample_weights[y_train == 0] = len(y_train) / (2 * (y_train == 0).sum())
                sample_weights[y_train == 1] = len(y_train) / (2 * (y_train == 1).sum())
                clf.fit(X_train, y_train, sample_weight=sample_weights)         #贝叶斯分类器
                # clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)  #XGB 和 lightGBM分类器
                # clf.fit(X_train, y_train)   #SVM分类器

                y_pred = clf.predict(X_val)
                y_prob = clf.predict_proba(X_val)[:, -1]

                ACC, Recall, Specificity, Precision, NPV, roc_auc = calculate_metrics(y_val, y_pred, y_prob)

                acc_scores.append(ACC)
                recall_scores.append(Recall)
                specificity_scores.append(Specificity)
                precision_scores.append(Precision)
                npv_scores.append(NPV)
                auc_scores.append(roc_auc)

                # # 更新指标历史
                # metrics_history['ACC'].append(np.mean(acc_scores))
                # metrics_history['Recall'].append(np.mean(recall_scores))
                # metrics_history['Specificity'].append(np.mean(specificity_scores))
                # metrics_history['Precision'].append(np.mean(precision_scores))
                # metrics_history['NPV'].append(np.mean(npv_scores))
                # metrics_history['AUC'].append(np.mean(auc_scores))
                for metric, scores in zip(
                    metrics_history.keys(),
                    [acc_scores, recall_scores, specificity_scores,
                     precision_scores, npv_scores, auc_scores]
                ):
                    metrics_history[metric].append(np.mean(scores))
            
            all_rounds_acc.append(np.mean(acc_scores))
            all_rounds_recall.append(np.mean(recall_scores))
            all_rounds_specificity.append(np.mean(specificity_scores))
            all_rounds_precision.append(np.mean(precision_scores))
            all_rounds_npv.append(np.mean(npv_scores))
            all_rounds_auc.append(np.mean(auc_scores))

            print(f"\n第 {round_num + 1} 轮五折交叉验证平均结果:")
            print(f"ACC:{np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
            print(f"Recall:{np.mean(recall_scores):.3f} ± {np.std(recall_scores):.3f}")
            print(f"Specificity:{np.mean(specificity_scores):.3f} ± {np.std(specificity_scores):.3f}")
            print(f"Precision:{np.mean(precision_scores):.3f} ± {np.std(precision_scores):.3f}")
            print(f"NPV:{np.mean(npv_scores):.3f} ± {np.std(npv_scores):.3f}")
            print(f"AUC:{np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")

            round_auc = np.mean(auc_scores)
            if round_auc > best_auc:
                best_auc = round_auc
                best_model = clf

        
        y_pred_final = best_model.predict(X_test_scaled)
        y_prob_final = best_model.predict_proba(X_test_scaled)[:, -1]
        final_ACC, final_Recall, final_Specificity, final_Precision, final_NPV, final_AUC = calculate_metrics(y_test_final, y_pred_final, y_prob_final)
        
        print("\n最终测试集的具体指标值:")
        print(f"准确率 (ACC): {final_ACC:.3f}")
        print(f"召回率 (Recall): {final_Recall:.3f}")
        print(f"特异性 (Specificity): {final_Specificity:.3f}")
        print(f"精确率 (Precision): {final_Precision:.3f}")
        print(f"阴性预测值 (NPV): {final_NPV:.3f}")
        print(f"AUC值: {final_AUC:.3f}")

        final_results = {
            'ACC': f"{final_ACC:.3f}",
            'Recall': f"{final_Recall:.3f}",
            'Specificity': f"{final_Specificity:.3f}",
            'Precision': f"{final_Precision:.3f}",
            'NPV': f"{final_NPV:.3f}",
            'AUC': f"{final_AUC:.3f}"
        }
        results = "五折交叉验证平均结果：\n"
        for metric, value in final_results.items():
            results += f"{metric}: {value}\n"
        save_results(results, result_folder)
        plot_metrics(metrics_history, result_folder)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()