import pandas as pd
import numpy as np

from plot_figures import Plot_Figures
from enums import similarity_method
from data_loader import Data_Loader



SIMILARITY_METHOD = similarity_method.MAPPED_RMSE



if __name__ == '__main__':
    # 加载保存的数据集
    df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
    print(f"✅ 数据集已加载：uninted_df.pkl")

    dl = Data_Loader(SIMILARITY_METHOD)
    da, rt, spread, feat = dl.transfer_data(df)
    scores = dl.cal_true_similarity_score(da, rt, spread)
    # scores.to_excel('mapped_scores.xlsx')
    # plot_spread_distribution(df)
    # plot_two_days_spread(spread, '2024-12-13', '2024-12-21')
    date_str1 = '2024-12-13'
    date_str2 = '2024-12-21'
    date_str1 = '2024-06-03'
    date_str2 = '2026-04-29'
    date_str1 = '2025-06-08'
    date_str2 = '2025-09-08'
    date_str1 = '2026-01-10'
    date_str2 = '2026-03-02'
    date_str1 = '2025-12-20'
    date_str2 = '2026-01-19'
    # 很不相似
    date_str1 = '2024-09-22'
    date_str2 = '2024-12-13'
    # date_str1 = '2024-06-01'
    # date_str2 = '2024-09-30'
    pf = Plot_Figures()
    pf.plot_two_days_spread_and_feat(spread, feat, date_str1, date_str2)
    a = 0