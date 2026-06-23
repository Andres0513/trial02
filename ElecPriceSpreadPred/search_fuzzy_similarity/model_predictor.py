import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import utils
import wide_table
from sklearn.model_selection import train_test_split

class Model_Predictor:
    def __init__(self, top_n=10):
        # ================= 模型模板 =================
        self.base_xgb_params = {
            "learning_rate": 0.08,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1,
            "reg_lambda": 1,
            "min_child_weight": 2,
            "random_state": 42,
            "n_jobs": -1,
            "enable_categorical": True,
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        }
        self.model = None
        self.top_n = top_n
        self.categorical_cols = utils.categorical_cols
        return

    def validate_model(self, x, true_y):
        pred_y = self.model.predict(x)
        rmse = np.sqrt(mean_squared_error(true_y, pred_y))
        return pred_y, rmse

    def run(self, train_df, test_df, spread):
        # train_df.drop(columns=['num_larger_mean', 'num_smaller_mean', 'num_cos_sim',  'week_day', 'season', 'is_holiday'], inplace=True)
        # test_df.drop(columns=['num_larger_mean', 'num_smaller_mean', 'num_cos_sim',  'week_day', 'season', 'is_holiday'], inplace=True)
        train_df.drop(
            columns=['num_larger_mean', 'num_smaller_mean', 'num_cos_sim'],
            inplace=True)
        test_df.drop(columns=['num_larger_mean', 'num_smaller_mean', 'num_cos_sim'],
                     inplace=True)
        # ================= 重采样 =================
        high_score_df = train_df[train_df['spread_cos'] > 0.75].copy()
        low_score_df = train_df[train_df['spread_cos'] < 0.25].copy()
        high_score_df_2 = train_df[train_df['spread_cos'] > 0.9].copy()
        low_score_df_2 = train_df[train_df['spread_cos'] < 0.1].copy()

        repeat_times = 20
        oversample_df_1 = pd.concat([high_score_df] * repeat_times, ignore_index=True)
        oversample_df_2 = pd.concat([low_score_df] * repeat_times, ignore_index=True)
        oversample_df_3 = pd.concat([high_score_df_2] * repeat_times, ignore_index=True)
        oversample_df_4 = pd.concat([low_score_df_2] * repeat_times, ignore_index=True)

        train_df = pd.concat([train_df, oversample_df_1], ignore_index=True)
        train_df = pd.concat([train_df, oversample_df_2], ignore_index=True)
        train_df = pd.concat([train_df, oversample_df_3], ignore_index=True)
        train_df = pd.concat([train_df, oversample_df_4], ignore_index=True)

        # ================= 日期预处理 =================
        train_df_backup = train_df.copy()  # 保留含target_date完整副本
        test_df_backup = test_df.copy()
        train_df = train_df.drop(columns=['target_date', 'reference_date'])
        test_df = test_df.drop(columns=['target_date', 'reference_date'])

        train_df[self.categorical_cols] = train_df[self.categorical_cols].astype('int').astype('category')
        test_df[self.categorical_cols] = test_df[self.categorical_cols].astype('int').astype('category')

        # ================= 划分输入输出 =================
        train_x = train_df.iloc[:, 1:]
        train_y_raw = train_df.iloc[:, 0]
        test_x = test_df.iloc[:, 1:]
        test_y_raw = test_df.iloc[:, 0]

        def map_target(x):
            return x

        train_y = train_y_raw.apply(map_target)
        test_y = test_y_raw.apply(map_target)

        # ================= 网格搜索参数组合 =================
        n_est_list = [20, 30, 40, 50, 100]
        max_depth_list = [4, 5, 6, 7, 8]
        all_candidates = []  # 存储全部参数组合，不再提前淘汰

        # 全局先算出日期掩码（只算一次，不用循环内重复计算）
        max_date = train_df_backup["target_date"].max()
        val_mask = train_df_backup["target_date"] == max_date
        train_mask = train_df_backup["target_date"] != max_date

        for n_est in n_est_list:
            for md in max_depth_list:
                curr_params = self.base_xgb_params.copy()
                curr_params["n_estimators"] = n_est
                curr_params["max_depth"] = md

                # 直接用掩码筛选原train_x/train_y，保留category dtype
                x_train_sub = train_x.loc[train_mask].copy()
                y_train_sub = train_y.loc[train_mask].copy()
                x_val_sub = train_x.loc[val_mask].copy()
                y_val_sub = train_y.loc[val_mask].copy()
                # 【修改：随机8:2拆分，替代日期掩码】
                x_train_sub, x_val_sub, y_train_sub, y_val_sub = train_test_split(
                    train_x, train_y, test_size=0.2, random_state=42, shuffle=True
                )

                temp_model = xgb.XGBRegressor(**curr_params)
                temp_model.fit(x_train_sub, y_train_sub)

                # 分别计算两段RMSE
                pred_train_sub = temp_model.predict(x_train_sub)
                pred_val_sub = temp_model.predict(x_val_sub)
                rmse_train_sub = np.sqrt(mean_squared_error(y_train_sub, pred_train_sub))
                rmse_val_sub = np.sqrt(mean_squared_error(y_val_sub, pred_val_sub))

                # 计算相对差值
                diff_abs = abs(rmse_train_sub - rmse_val_sub)
                mean_rmse = (rmse_train_sub + rmse_val_sub) / 2
                relative_diff = diff_abs / mean_rmse if mean_rmse != 0 else 0

                print(
                    f"n_estimators={n_est}, max_depth={md} | "
                    f"历史日期训练RMSE={rmse_train_sub:.4f}, "
                    f"最大日期验证RMSE={rmse_val_sub:.4f}, "
                    f"相对差值={relative_diff:.2%}"
                )
                if relative_diff > 1.25:
                    print(f"  两段RMSE相对差值超过25%，该参数组合过拟合严重\n")

                # 全部存入候选列表，不提前淘汰
                all_candidates.append({
                    "params": curr_params,
                    "train_sub_rmse": rmse_train_sub,
                    "relative_diff": relative_diff
                })

        # 分场景选择最优模型
        valid_candidates = [item for item in all_candidates if item["relative_diff"] <= 1.25]
        if valid_candidates:
            # 存在差值≤25%合格模型：选训练集RMSE最小
            best_item = min(valid_candidates, key=lambda d: d["train_sub_rmse"])
            print(f"\n===== 筛选完成：存在差值≤25%的合格模型 =====")
        else:
            # 全部超阈值：选取相对差值最小的兜底，不抛异常
            print(f"\n===== 警告：所有参数组合差值均超过25%，自动选取差值最小的模型兜底运行 =====")
            best_item = min(all_candidates, key=lambda d: d["relative_diff"])

        best_params = best_item["params"]
        print(f"最终选用参数：{best_params}")
        print(f"训练子集RMSE={best_item['train_sub_rmse']:.6f}，训练/验证相对差值={best_item['relative_diff']:.2%}\n")

        # 使用最优参数，在完整原始训练集上重新训练，再跑真实测试集
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(train_x, train_y)

        # ================= 评估与后处理（原逻辑完全不变） =================
        pred_y_train, rmse_train = self.validate_model(train_x, train_y)
        pred_y_test, rmse_test = self.validate_model(test_x, test_y)

        test_res = pd.concat([test_df_backup, pd.DataFrame(pred_y_test, columns=['pred_y'])], axis=1)
        cols = ['pred_y'] + [c for c in test_res.columns if c != 'pred_y']
        test_res = test_res[cols]

        test_res = utils.remove_same_date_then_sort(test_res, 'pred_y')
        final_test_res = utils.cal_spread(test_res, spread, self.top_n)

        wt = wide_table.Wide_Table()
        wide_test_res = wt.run(final_test_res, spread, test_res)


        tmp = wide_test_res.merge(test_df_backup, on=['target_date', 'reference_date'], how='left')
        return wide_test_res