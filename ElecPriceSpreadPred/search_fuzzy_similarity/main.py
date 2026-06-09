import pandas as pd
from fuzzy_similarity_data import Fuzzy_Similarity_Data
from model_predictor import Model_Predictor
from datetime import date

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
    # train_df, val_df, test_df = split_train_and_test(all_df)
    # 定义起始/结束日期
    start_date = date(2026, 2, 1)
    end_date = date(2026, 5, 14)

    # 生成日期范围并转为date对象列表
    date_list = pd.date_range(start=start_date, end=end_date).date.tolist()

    a = 0