import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import utils
import wide_table

class Model_Predictor02:
    def __init__(self, top_n=10):
        # ================= 模型 =================
        self.model = xgb.XGBRegressor(
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
        self.top_n = 10
        self.categorical_cols = utils.categorical_cols
        return

    def validate_model(self, x, true_y):
        pred_y = self.model.predict(x)
        rmse = np.sqrt(mean_squared_error(true_y, pred_y))
        return pred_y, rmse

    def run(self, train_df, test_df, spread):
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
        test_df_backup = test_df.copy()
        train_df = train_df.drop(columns=['target_date', 'reference_date'])
        test_df = test_df.drop(columns=['target_date', 'reference_date'])

        # 转 category（XGBoost 支持直接训练）
        train_df[self.categorical_cols] = train_df[self.categorical_cols].astype('int').astype('category')
        test_df[self.categorical_cols] = test_df[self.categorical_cols].astype('int').astype('category')

        # ================= 划分输入输出 =================
        train_x = train_df.iloc[:, 1:]
        train_y = train_df.iloc[:, 0]
        test_x = test_df.iloc[:, 1:]
        test_y = test_df.iloc[:, 0]

        # 训练
        self.model.fit(train_x, train_y)
        # 训练集
        pred_y_train, rmse_train = self.validate_model(train_x, train_y)
        # 测试集

        pred_y_test, rmse_test = self.validate_model(test_x, test_y)

        test_res = pd.concat([test_df_backup, pd.DataFrame(pred_y_test, columns=['pred_y'])], axis=1)
        # 把 pred_y 移到第一列
        cols = ['pred_y'] + [c for c in test_res.columns if c != 'pred_y']
        test_res = test_res[cols]

        test_res = utils.remove_same_date_then_sort(test_res, 'pred_y')
        final_test_res = utils.cal_spread(test_res, spread, self.top_n)

        wt = wide_table.Wide_Table()
        wide_test_res = wt.run(final_test_res, spread, test_res)

        return wide_test_res