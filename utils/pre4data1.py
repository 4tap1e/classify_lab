import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, roc_auc_score

def main():
    data_path = 'D:/PycharmProject/classification/t0_n1_2.xlsx'
    output_path = 'D:/PycharmProject/classification/t0_train0.xlsx'

    data = pd.read_excel(data_path, engine='openpyxl')
    try:
        # df = pd.read_excel(output_file, engine='openpyxl')
        # df = pd.read_excel(data_path, engine='openpyxl')
        # df1 = pd.read_excel(data1_path, engine='openpyxl')
        # print(f"t0_n1_2.xlsx的形状: {df.shape}")
        # print(f"t0_n1_1.xlsx的形状: {df1.shape}")
        # df.drop('sheet_source', axis=1, inplace=Tr ue) 
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # 初始化 XGBoost 分类器
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            # use_label_encoder=False,
            random_state=42
        )
        feature_selector = SelectFromModel(estimator=xgb_clf, max_features=100, prefit=False)

        pipeline = Pipeline([
            ('feature_selection', feature_selector),
            ('classifier', xgb_clf)
        ])
        
        param_grid = {
            'feature_selection__estimator__n_estimators': [100, 200],
            'classifier__max_depth': [4, 6, 8],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.7, 0.8, 0.9],
            'classifier__colsample_bytree': [0.7, 0.8, 0.9]
        }
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,  # 五折交叉验证
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

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        print("分类报告：\n", classification_report(y_test, y_pred))
        print("ROC-AUC 分数：", roc_auc_score(y_test, y_pred_proba))

        selected_features = best_model.named_steps['feature_selection'].get_support(indices=True)
        selected_features_names = X.columns[selected_features]

        importances = best_model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_importance_df = pd.DataFrame({
            '特征': selected_features_names,
            '重要性': importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='重要性', ascending=False)
        feature_importance_df.to_excel(output_path, index=False)

    except Exception as e:
        print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    main()
