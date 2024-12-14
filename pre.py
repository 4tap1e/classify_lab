import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pre4data import lasso_dimension_reduction, analyze_columns, CPC_change, get_importantfeature_xgb, drop_columns

def save_splits(X_train, X_val, y_train, y_val, output_dir):
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 合并特征和标签
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        train_path = os.path.join(output_dir, 'train_set.xlsx')
        val_path = os.path.join(output_dir, 'val_set.xlsx')
        
        train_df.to_excel(train_path, index=False)
        val_df.to_excel(val_path, index=False)
        
    except Exception as e:
        print(f"保存数据集时出错: {str(e)}")

def clean(data):
    cols_to_drop = [col for col in data.columns if data[col].nunique() == 1]
    data = data.drop(columns=cols_to_drop)
    print(f" 删除了 {len(cols_to_drop)} 列")
    
    return data

def main():
    data_path = 'D:/PycharmProject/classification/t1_s123.xlsx' # E:/CTSeg/code-zhc/time0.xlsx  D:/PycharmProject/classification/time0.xlsx
    # output_test_path = 'D:/PycharmProject/classification/t1_s3_test_lasso.xlsx'
    # output_train_path = 'D:/PycharmProject/classification/t1_s3_train_lasso.xlsx'
    output_train_path = 'D:/PycharmProject/classification/t1_s123_train.xlsx'

    dropdata012 = ['diagnostics_Image-original_Hash', 'diagnostics_Mask-original_Hash',
                'diagnostics_Image-original_Spacing','diagnostics_Image-original_Size',
                'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_Size',
                'diagnostics_Mask-original_BoundingBox','diagnostics_Mask-original_CenterOfMassIndex',
                'diagnostics_Mask-original_CenterOfMass']
    dropdata3 = ['diagnostics_Image-original_Hash', 'diagnostics_Image-original_Hash_1', 'diagnostics_Image-original_Hash_2', 'diagnostics_Image-original_Hash_3',
                 'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Hash_1', 'diagnostics_Mask-original_Hash_2', 'diagnostics_Mask-original_Hash_3',
                'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Spacing_1', 'diagnostics_Image-original_Spacing_2','diagnostics_Image-original_Spacing_3',
                'diagnostics_Image-original_Size', 'diagnostics_Image-original_Size_1', 'diagnostics_Image-original_Size_2', 'diagnostics_Image-original_Size_3',
                'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Spacing_1', 'diagnostics_Mask-original_Spacing_2', 'diagnostics_Mask-original_Spacing_3',
                'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_Size_1', 'diagnostics_Mask-original_Size_2', 'diagnostics_Mask-original_Size_3',
                'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_BoundingBox_1', 'diagnostics_Mask-original_BoundingBox_2', 'diagnostics_Mask-original_BoundingBox_3',
                'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMassIndex_1', 'diagnostics_Mask-original_CenterOfMassIndex_2', 'diagnostics_Mask-original_CenterOfMassIndex_3',
                'diagnostics_Mask-original_CenterOfMass', 'diagnostics_Mask-original_CenterOfMass_1', 'diagnostics_Mask-original_CenterOfMass_2', 'diagnostics_Mask-original_CenterOfMass_3',
                'diagnostics_Mask-original_BoundingBox.1', 'diagnostics_Mask-original_BoundingBox.1_1', 'diagnostics_Mask-original_BoundingBox.1_2', 'diagnostics_Mask-original_BoundingBox.1_3',  
                'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_BoundingBox_1', 'diagnostics_Mask-original_BoundingBox_2', 'diagnostics_Mask-original_BoundingBox_3',
                'CPC_1', 'CPC_2', 'CPC_3'] 
    
    data = pd.read_excel(data_path)
    data = drop_columns(data, dropdata3)
    cols = [col for col in data.columns if col != 'CPC'] + ['CPC']
    data = data[cols]
    data = CPC_change(data)
    same_cols, variance_cols = analyze_columns(data, var_threshold=5)
    # data = data.drop(columns=variance_cols)
    data = data.drop(columns=same_cols.keys())
    print(f"data.shape: {data.shape}")

    # train_size = 0.9
    # data_train, data_test = train_test_split(data, train_size=train_size, random_state=42,
    #                                                               shuffle=True, stratify=data['label'])
    
    imputer = SimpleImputer(strategy='mean')
    feature_names = data.columns.tolist()
    data_train_imputed = imputer.fit_transform(data)
    # data_test_imputed = imputer.transform(data_test)
    data_train = pd.DataFrame(data_train_imputed, columns=feature_names)
    # data_test = pd.DataFrame(data_test_imputed, columns=feature_names)

    data_train.to_excel(output_train_path, index=False)

    train_counts = data_train['label'].value_counts()
    train_proportions = data_train['label'].value_counts(normalize=True)
    # test_counts = data_test['label'].value_counts()
    # test_proportions = data_test['label'].value_counts(normalize=True)
    print("\nData_train - Counts:")
    print(train_counts)
    print("\nData_train - Proportions:")
    print(train_proportions)
    # print("\nData_test - Counts:")
    # print(test_counts)
    # print("\nData_test - Proportions:")
    # print(test_proportions)
    # print(f"data_test.shape: {data_test.shape}, data_train.shape: {data_train.shape}")


    # imputer = SimpleImputer(strategy='mean')
    # feature_names = data_train.columns.tolist()
    # data_train_imputed = imputer.fit_transform(data_train)
    # data_test_imputed = imputer.transform(data_test)
    # data_train = pd.DataFrame(data_train_imputed, columns=feature_names)
    # data_test = pd.DataFrame(data_test_imputed, columns=feature_names)

    # # # 使用Lasso进行特征选择
    # X_train_lasso, selected_features = lasso_dimension_reduction(data_train, cv=5, n_alphas=1000, alpha_range=(-7, -2))
    # X_test_lasso = data_test[selected_features + ['label']]
    # print(f"X_train_lasso.shape: {X_train_lasso.shape}")
    # X_train_lasso.to_excel(output_train_path, index=False)
    # X_test_lasso.to_excel(output_test_path, index=False)
    # print(f"X_test_lasso.shape: {X_test_lasso.shape}")

    # # 使用XGBoost进行特征选择
    # X_train_xgb, selected_features_names = get_importantfeature_xgb(data_train)
    # X_test_xgb = data_test[selected_features_names].copy()
    # X_test_xgb['label'] = data_test['label']
    # print(f"X_train_xgb.shape: {X_train_xgb.shape}, X_test_xgb.shape: {X_test_xgb.shape}")
    # X_train_xgb.to_excel(output_train1_path, index=False)
    # X_test_xgb.to_excel(output_test1_path, index=False)

if __name__ == "__main__":
    main()