import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib

# 数据路径 (用户指定的全局路径)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from htgy.globals import *

# 读取数据
data_path = PROCESSED_DIR / "Vege_Input_Data_Updated_copy.csv"
df = pd.read_csv(data_path)

# 选择关键列
lai_col = "2007 LAI"
soc_col = "SOC Monthly Increase (g/kg/month)"

# 数据清理函数
def clean_numeric(value):
    try:
        return float(str(value).replace('−', '-').replace('m', '').replace(' ', '').replace(',', ''))
    except:
        return np.nan

# 清理数据
for col in [lai_col, soc_col]:
    df[col] = df[col].apply(clean_numeric)

# 删除缺失值
df_cleaned = df[[lai_col, soc_col]].dropna()
df_cleaned = df_cleaned[(df_cleaned[soc_col] >= -0.1) & (df_cleaned[soc_col] <= 0.4)]

# 定义特征X和目标Y
X = df_cleaned[[lai_col]].values
Y = df_cleaned[soc_col].values

# 数据划分（80%训练集，20%测试集）
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# XGBoost 模型训练
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, Y_train)

# 保存模型
model_filename = 'lai_soc_xgb_model.pkl'
joblib.dump(xgb_model, model_filename)

# 模型预测
Y_pred = xgb_model.predict(X_test)

# 模型评估
r2 = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print(f"XGBoost 模型的决定系数 (R²): {r2:.4f}")
print(f"XGBoost 模型的均方根误差 (RMSE): {rmse:.4f}")

# 误差分析可视化
plt.figure(figsize=(12,5))

# 实际值 vs 预测值
plt.subplot(1, 2, 1)
sns.scatterplot(x=Y_test, y=Y_pred)
plt.xlabel('Observed SOC Increase (g/kg/month)')
plt.ylabel('Predicted SOC Increase (g/kg/month)')
plt.title('Observed vs Predicted SOC')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')

# 残差分布
plt.subplot(1, 2, 2)
residuals = Y_test - Y_pred
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals (Observed - Predicted)')
plt.title('Residuals Distribution')

plt.tight_layout()
plt.show()

# 特征重要性可视化
xgb.plot_importance(xgb_model)
plt.title('XGBoost Feature Importance')
plt.show()
