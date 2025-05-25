from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np

# 假设 features, targets 列表已定义，df 是总表
results = []

for use_pca in [False, True]:
    errors = []
    for target in targets:
        df_target = df.dropna(subset=[target] + features)
        X = df_target[features].values
        y = df_target[target].values
        if use_pca:
            pca = PCA(n_components=0.95)  # 保留95%方差
            X = pca.fit_transform(X)
        # 你用的模型
        model = get_model('lr')  # 或 'svm', 'lwr'
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = mean_squared_error(y_test, y_pred, squared=False)
        errors.append(error)
    avg_error = np.mean(errors)
    results.append({'use_pca': use_pca, 'avg_rmse': avg_error})

print(results)