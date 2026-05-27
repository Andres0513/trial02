import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from enums import similarity_method

class Data_Loader:
    def __init__(self, SIMILARITY_METHOD):
        self.SIMILARITY_METHOD = SIMILARITY_METHOD

    def transfer_data(self, df):
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

    def cal_true_similarity_score(self, da, rt, spread):
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

        level_data_matrix = spread.values
        if self.SIMILARITY_METHOD == similarity_method.MAPPED_RMSE:
            spread_level = spread.map(map_spread_to_level)
            level_data_matrix = spread_level.values

        for i in range(0, len(data_matrix)):
            for j in range(i + 1, len(data_matrix)):
                values1 = data_matrix[i]
                values2 = data_matrix[j]
                score = 1e6
                if self.SIMILARITY_METHOD == similarity_method.COSINE:
                    score = cosine_similarity(values1.reshape(1, -1), values2.reshape(1, -1))[0][0]
                    score = (score+1)/2
                if self.SIMILARITY_METHOD == similarity_method.RMSE:
                    score = np.sqrt(np.mean((values1 - values2) ** 2))
                if self.SIMILARITY_METHOD == similarity_method.PEARSON:
                    score = pearsonr(values1, values2)
                if self.SIMILARITY_METHOD == similarity_method.MAPPED_RMSE:
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
