import pandas as pd
from enums import similarity_method
from fuzzy_similarity_data import Fuzzy_Similarity_Data
from datetime import date, timedelta, time
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from wide_table import Wide_Table

import utils


def split_train_and_test(all_df):
    val_split_date = date(2026,4,28)
    test_split_date = date(2026,5,1)
    train_df = all_df[all_df['target_date'] < val_split_date]
    val_df = all_df[(all_df['target_date'] >= val_split_date) & (all_df['target_date'] < test_split_date)]
    test_df = all_df[all_df['target_date'] >= test_split_date]
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df

def validate_model(model, x, true_y):
    pred_y = model.predict(x)
    # 计算指标
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    return pred_y, rmse


if __name__ == '__main__':
    # 加载保存的数据集
    # df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
    df = pd.read_csv("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.csv", index_col=0)
    # df = pd.read_csv("/Users/bytedance/Codes/Python/e/ElecPriceSpreadPred/uninted_df.csv", index_col=0)
    print(f"✅ 数据集已加载：uninted_df.pkl")

    fsd = Fuzzy_Similarity_Data()
    # 需要的数据日期范围
    date_range = [['2024-05-02', '2026-05-02']]
    spread, all_df = fsd.run(df, date_range)
    spread = spread.reset_index()
    train_df, val_df, test_df = split_train_and_test(all_df)
    #  ================= 数据分布 =================
    # 5% 分位数
    q5 = np.percentile(train_df['spread_cos'], 5)
    # 95% 分位数
    q95 = np.percentile(train_df['spread_cos'], 95)

    print(f"5% 分位数: {q5:.5f}")
    print(f"95% 分位数: {q95:.5f}")

    # ================= 重采样 =================
    # 1. 选出要增强的样本
    high_score_df = train_df[train_df['spread_cos'] > 0.75].copy()
    low_score_df = train_df[train_df['spread_cos'] < 0.25].copy()
    high_score_df_2 = train_df[train_df['spread_cos'] > 0.9].copy()
    low_score_df_2 = train_df[train_df['spread_cos'] < 0.1].copy()

    # 2. 每个样本复制 20 次
    repeat_times = 20
    oversample_df_1 = pd.concat([high_score_df] * repeat_times, ignore_index=True)
    oversample_df_2 = pd.concat([low_score_df] * repeat_times, ignore_index=True)
    oversample_df_3 = pd.concat([high_score_df_2] * repeat_times, ignore_index=True)
    oversample_df_4 = pd.concat([low_score_df_2] * repeat_times, ignore_index=True)

    # 3. 追加回训练集
    train_df = pd.concat([train_df, oversample_df_1], ignore_index=True)
    train_df = pd.concat([train_df, oversample_df_2], ignore_index=True)
    train_df = pd.concat([train_df, oversample_df_3], ignore_index=True)
    train_df = pd.concat([train_df, oversample_df_4], ignore_index=True)

    # ================= 对日期进行预处理 =================
    train_df_backup = train_df.copy()
    val_df_backup = val_df.copy()
    test_df_backup = test_df.copy()
    train_df = train_df.drop(columns=['target_date', 'reference_date'])
    val_df = val_df.drop(columns=['target_date', 'reference_date'])
    test_df = test_df.drop(columns=['target_date', 'reference_date'])

    # 枚举特征：周几、季度、电网工况 → 转成 category 类型
    categorical_cols = []
    for col in train_df.columns:
        if 'grid_env' in col or 'week_day' in col or 'season' in col or 'is_holiday' in col:
            categorical_cols.append(col)

    # 转 category（XGBoost 支持直接训练）
    train_df[categorical_cols] = train_df[categorical_cols].astype('int').astype('category')
    val_df[categorical_cols] = val_df[categorical_cols].astype('int').astype('category')
    test_df[categorical_cols] = test_df[categorical_cols].astype('int').astype('category')

    # ================= 划分输入输出 =================
    train_x = train_df.iloc[:, 0:]
    train_y = train_df.iloc[:, 0]
    val_x = val_df.iloc[:, 0:]
    val_y = val_df.iloc[:, 0]
    test_x = test_df.iloc[:, 0:]
    test_y = test_df.iloc[:, 0]

    # 模型
    model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=1,
        min_child_weight=2,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        objective="reg:squarederror",  # 回归损失
        eval_metric="rmse"  # 回归常用评价：rmse / mae
    )

    # 训练
    model.fit(train_x, train_y)
    # 训练集
    pred_y_train, rmse_train = validate_model(model, train_x, train_y)
    # 验证集
    pred_y_val, rmse_val = validate_model(model, val_x, val_y)
    # 测试集
    pred_y_test, rmse_test = validate_model(model, test_x, test_y)

    val_res = pd.concat([val_df_backup, pd.DataFrame(pred_y_val, columns=['pred_y'])], axis=1)
    # 把 pred_y 移到第一列
    cols = ['pred_y'] + [c for c in val_res.columns if c != 'pred_y']
    val_res = val_res[cols]

    test_res = pd.concat([test_df_backup, pd.DataFrame(pred_y_test, columns=['pred_y'])], axis=1)
    # 把 pred_y 移到第一列
    cols = ['pred_y'] + [c for c in test_res.columns if c != 'pred_y']
    test_res = test_res[cols]

    val_res = utils.sort_df(val_res, 'pred_y')
    final_val_res = utils.cal_spread(val_res, spread, 10)

    test_res = utils.sort_df(test_res, 'pred_y')
    final_test_res = utils.cal_spread(test_res, spread, 10)

    wt = Wide_Table()
    wide_val_res = wt.run(final_val_res, spread, val_res)

    wide_test_res = wt.run(final_test_res, spread, test_res)

