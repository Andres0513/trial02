import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from plot_figures import Plot_Figures
from enums import similarity_method



SIMILARITY_METHOD = similarity_method.MAPPED_RMSE

def transfer_data(df):
    tmp = df.copy()
    tmp['日期'] = pd.to_datetime(tmp['时间']).dt.date

    d = tmp[['时间', '日前价格', '实时价格', '价差（实时-日前）']].copy()
    d['时间'] = pd.to_datetime(d['时间'])
    d['日期'] = d['时间'].dt.date
    d['时刻'] = d['时间'].dt.time

    # ============ 按日期获取特征 ============
    days_cols = [col for col in tmp.columns if 'day' in col.lower()]
    df_feat = tmp[['日期'] + days_cols + ['grid_env', '星期', '季度', '是否节假日']].copy()
    feat = df_feat.groupby('日期').first()

    # 生成 3 个独立DF（每行=日期，每列=00:00~23:30）
    da = d.pivot_table(index='日期', columns='时刻', values='日前价格', aggfunc='first')
    rt = d.pivot_table(index='日期', columns='时刻', values='实时价格', aggfunc='first')
    spread = d.pivot_table(index='日期', columns='时刻', values='价差（实时-日前）', aggfunc='first')

    # =============只要任意一个表有 NA，整行删除================
    mask_da = da.isna().any(axis=1)
    mask_rt = rt.isna().any(axis=1)
    mask_spread = spread.isna().any(axis=1)
    mask_feat = feat.isna().any(axis=1)
    bad_rows = mask_da | mask_rt | mask_spread | mask_feat
    # 三张表 同时删除坏行
    da = da[~bad_rows]
    rt = rt[~bad_rows]
    spread = spread[~bad_rows]
    feat = feat[~bad_rows]

    return da, rt, spread, feat

def cal_true_similarity_score(da, rt, spread):
    res = []
    dates = spread.index.to_list()
    data_matrix = spread.values

    def map_spread_to_level(x):
        if pd.isna(x):
            return None
        if x < -110:
            return -5
        elif -110 <= x < -32:
            return -4
        elif -32 <= x < -15:
            return -3
        elif -15 <= x < -5:
            return -2
        elif -5 <= x < 0:
            return -1
        elif 0 <= x < 5:
            return 1
        elif 5 <= x < 15:
            return 2
        elif 15 <= x < 32:
            return 3
        elif 32 <= x < 110:
            return 4
        else:  # >= 100
            return 5

    if SIMILARITY_METHOD == similarity_method.MAPPED_RMSE:
        spread_level = spread.map(map_spread_to_level)
        level_data_matrix = spread_level.values

    for i in range(0, len(data_matrix)):
        for j in range(i + 1, len(data_matrix)):
            values1 = data_matrix[i]
            values2 = data_matrix[j]
            if SIMILARITY_METHOD == similarity_method.COSINE:
                score = cosine_similarity(values1.reshape(1, -1), values2.reshape(1, -1))[0][0]
                score = (score+1)/2
            if SIMILARITY_METHOD == similarity_method.RMSE:
                score = np.sqrt(np.mean((values1 - values2) ** 2))
            if SIMILARITY_METHOD == similarity_method.PEARSON:
                score = pearsonr(values1, values2)
            if SIMILARITY_METHOD == similarity_method.MAPPED_RMSE:
                values1 = level_data_matrix[i]
                values2 = level_data_matrix[j]
                score = np.sqrt(np.mean((values1 - values2) ** 2))
            res.append(
                {
                    'date1': dates[i],
                    'date2': dates[j],
                    'similarity_score': score
                }
            )

    res = pd.DataFrame(res)
    return res


if __name__ == '__main__':
    # 加载保存的数据集
    df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
    print(f"✅ 数据集已加载：uninted_df.pkl")
    da, rt, spread, feat = transfer_data(df)
    scores = cal_true_similarity_score(da, rt, spread)
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