import pandas as pd
from fuzzy_similarity_data import Fuzzy_Similarity_Data
from model_predictor import Model_Predictor
from model_predictor_02 import Model_Predictor02
from datetime import date
import utils

def split_train_and_test_backup(all_df, test_date, forward_days):
    train_df = all_df[(pd.to_datetime(all_df['target_date']) < pd.to_datetime(test_date)) &
                      (pd.to_datetime(all_df['target_date']) >= pd.to_datetime(test_date) - pd.Timedelta(days=forward_days))]
    test_df = all_df[pd.to_datetime(all_df['target_date']) == pd.to_datetime(test_date)]
    # 遍历枚举列，如果test_df中的枚举值在trian_df中不存在，则
    # 1.如果all_df中能找到target_date<test_date-forward_days的，且枚举列等于该枚举值的行，则取距离test_date最近的14天的行，将其添加到train_df中，不足14天的，有多少取多少
    # 2.如果没有找到，则将test_df中的枚举值设为train_df中该列的最多值
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

def split_train_and_test(all_df, test_date, forward_days):
    # 统一转datetime
    all_df = all_df.copy()
    all_df['target_date'] = pd.to_datetime(all_df['target_date'])
    test_date_dt = pd.to_datetime(test_date)
    train_start_date = test_date_dt - pd.Timedelta(days=forward_days)

    # ---------------------- 1. 基础切分（天粒度，保留所有行） ----------------------
    train_df = all_df[
        (all_df['target_date'] < test_date_dt) &
        (all_df['target_date'] >= train_start_date)
        ].copy()

    test_df = all_df[
        all_df['target_date'] == test_date_dt
        ].copy()

    # ---------------------- 2. 处理枚举值缺失（按日期补，不是按行补） ----------------------
    for col in utils.categorical_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        train_unique = set(train_df[col].dropna())
        test_unique = set(test_df[col].dropna())
        missing_vals = test_unique - train_unique

        for val in missing_vals:
            # ---- 情况1：去更早历史找：target_date < train_start_date，且 col == val ----
            history_candidate = all_df[
                (all_df['target_date'] < train_start_date) &
                (all_df[col] == val)
                ]
            if not history_candidate.empty:
                # 取【最近最多14个日期】（先去重日期、排序、取14天、再拿所有行）
                history_dates = history_candidate['target_date'].dropna().unique()
                history_dates = sorted(history_dates, reverse=True)[:forward_days]  # 最近14天
                add_data = history_candidate[history_candidate['target_date'].isin(history_dates)]
                train_df = pd.concat([train_df, add_data], ignore_index=True)

            # ---- 情况2：完全没历史 → test_df 该值替换为训练集众数 ----
            else:
                if train_df[col].notna().any():
                    mode_val = train_df[col].mode()[0]
                    test_df.loc[test_df[col] == val, col] = mode_val
    train_df['target_date'] = train_df['target_date'].dt.date
    test_df['target_date'] = test_df['target_date'].dt.date
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


if __name__ == '__main__':
    # # ================= 加载原始数据并加工后保存 =================
    # # 加载保存的数据集
    # # df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
    # df = pd.read_csv("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.csv", index_col=0)
    # # df = pd.read_csv("/Users/bytedance/Codes/Python/e/ElecPriceSpreadPred/uninted_df.csv", index_col=0)
    # print(f"✅ 数据集已加载：uninted_df.pkl")
    #
    # fsd = Fuzzy_Similarity_Data()
    # # 需要的数据日期范围
    # date_range = [['2024-05-02', '2026-06-12']]
    # spread, all_df = fsd.run(df, date_range, demo=False)    # demo=False 时，返回所有数据
    # spread.to_csv("spread.csv", index=True)
    # all_df.to_csv("all_df.csv", index=False)
    # ================= 加载保存的加工好的数据集 =================
    # 1. 加载
    spread = pd.read_csv("spread.csv")
    all_df = pd.read_csv("all_df.csv")
    spread = spread.reset_index(drop=True)
    all_df = all_df.reset_index(drop=True)
    # 2. 处理日期格式
    spread['日期'] = pd.to_datetime(spread['日期']).dt.date
    all_df['target_date'] = pd.to_datetime(all_df['target_date']).dt.date
    all_df['reference_date'] = pd.to_datetime(all_df['reference_date']).dt.date

    # train_df, val_df, test_df = split_train_and_test(all_df)
    # 定义起始/结束日期
    start_date = date(2026, 4, 29)
    end_date = date(2026, 4, 29)

    # 生成日期范围并转为date对象列表
    date_list = pd.date_range(start=start_date, end=end_date).date.tolist()

    mp = Model_Predictor(top_n=10)
    forward_days = 2   # 使用前几天数据进行训练
    pred_res = []
    all_res = []
    for test_date in date_list:
        print(f"current date: {test_date}")
        # train_df, test_df = split_train_and_test_backup(all_df, test_date, forward_days)
        train_df, test_df = split_train_and_test(all_df, test_date, forward_days)
        wide_test_res = mp.run(train_df, test_df, spread)
        all_res.append(wide_test_res)
        pred_res.append(wide_test_res.loc[wide_test_res['reference_date'].isin(['raw_spread', 'pred_spread']), :])
    pred_res_df = pd.concat(pred_res, axis=0)
    all_res_df = pd.concat(all_res, axis=0)
    pred_res_df.drop(columns=['pred_y', 'spread_cos'], inplace=True)
    pred_res_df.to_csv("pred_res_df.csv", index=False)
    all_res_df.to_csv("all_res_df.csv", index=False)
    print("✅ 所有程序运行完毕！")
