import pandas as pd
from fuzzy_similarity_data import Fuzzy_Similarity_Data
from model_predictor import Model_Predictor
from datetime import date

def split_train_and_test(all_df, test_date, forward_days):
    train_df = all_df[(pd.to_datetime(all_df['target_date']) < pd.to_datetime(test_date)) &
                      (pd.to_datetime(all_df['target_date']) >= pd.to_datetime(test_date) - pd.Timedelta(days=forward_days))]
    test_df = all_df[pd.to_datetime(all_df['target_date']) == pd.to_datetime(test_date)]
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


if __name__ == '__main__':
    # # ================= 数据 =================
    # # 加载保存的数据集
    # # df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
    # df = pd.read_csv("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.csv", index_col=0)
    # # df = pd.read_csv("/Users/bytedance/Codes/Python/e/ElecPriceSpreadPred/uninted_df.csv", index_col=0)
    # print(f"✅ 数据集已加载：uninted_df.pkl")
    #
    # fsd = Fuzzy_Similarity_Data()
    # # 需要的数据日期范围
    # date_range = [['2024-05-02', '2026-05-02']]
    # spread, all_df = fsd.run(df, date_range, demo=False)    # demo=False 时，返回所有数据
    # spread.to_csv("spread.csv", index=True)
    # all_df.to_csv("all_df.csv", index=False)
    # ================= 加载保存的数据集 =================
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
    start_date = date(2026, 2, 1)
    end_date = date(2026, 5, 14)

    # 生成日期范围并转为date对象列表
    date_list = pd.date_range(start=start_date, end=end_date).date.tolist()

    mp = Model_Predictor(top_n=10)
    forward_days = 14   # 使用前几天数据进行训练
    pred_res = []
    all_res = []
    for test_date in date_list:
        print(f"current date: {test_date}")
        train_df, test_df = split_train_and_test(all_df, test_date, forward_days)
        wide_test_res = mp.run(train_df, test_df, spread)
        all_res.append(wide_test_res)
        pred_res.append(wide_test_res.iloc[wide_test_res['reference_date'].isin(['raw_spread', 'pred_spread']), :])
    pred_res_df = pd.concat(pred_res, axis=0)
    all_res_df = pd.concat(all_res, axis=0)
    a = 0
