import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Fuzzy_Similarity_Data:
    def __init__(self):
        return

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

    def extract_specific_feat(self, feat):
        feat_dict = {}
        for date in feat.index:
            feat_dict[date] = {
                'bid': feat.loc[date, [f'day_日前竞价空间{i}' for i in range(96)]].values.astype(np.float32),
                'load': feat.loc[date, [f'day_统调负荷预测{i}' for i in range(96)]].values.astype(np.float32),
                'imported': feat.loc[date, [f'day_外来电计划{i}' for i in range(96)]].values.astype(np.float32),
                'pv': feat.loc[date, [f'day_光伏出力预测{i}' for i in range(96)]].values.astype(np.float32),
                'wind_power': feat.loc[date, [f'day_风电出力预测{i}' for i in range(96)]].values.astype(np.float32),
                'fixed': feat.loc[date, [f'day_固定出力计划{i}' for i in range(96)]].values.astype(np.float32),
                'temp': feat.loc[date, [f'day_温度{i}' for i in range(24)]].values.astype(np.float32),
                'humidity': feat.loc[date, [f'day_湿度{i}' for i in range(24)]].values.astype(np.float32),
                'rain': feat.loc[date, [f'day_降雨{i}' for i in range(24)]].values.astype(np.float32),
                'irr': feat.loc[date, [f'day_辐照{i}' for i in range(24)]].values.astype(np.float32),
                'cloud': feat.loc[date, [f'day_云{i}' for i in range(24)]].values.astype(np.float32),
                'wind': feat.loc[date, [f'day_风速{i}' for i in range(24)]].values.astype(np.float32),
                'num': feat.loc[date, ['grid_env', '星期', '季度', '是否节假日']].values.astype(np.float32)
            }
        return feat_dict

    def cal_pair_feat(self, feat_1_nda, feat_2_nda):
        mean_1 = feat_1_nda.mean()
        mean_2 = feat_2_nda.mean()
        cos_sim = cosine_similarity(feat_1_nda.reshape(1, -1), feat_2_nda.reshape(1, -1))[0][0]
        return mean_1, mean_2, cos_sim

    def cal_all_train_df(self, spread, feat):
        feat_dict = self.extract_specific_feat(feat)
        len_date = len(feat)
        res = []
        for i in range(len_date-14 ,len_date): # 目标日期（前两周内）
            for j in range(0, i+1): # 参考日期
                larger_date = feat.index[i]
                smaller_date = feat.index[j]
                feat_larger_data = feat_dict.get(larger_date)
                feat_smaller_data = feat_dict.get(smaller_date)
                one_row_dict = {}
                one_row_dict["target_date"] = larger_date
                one_row_dict["reference_date"] = smaller_date
                # ========================= 计算两个日期的价差相似 =========================
                _, _, spread_cos = self.cal_pair_feat(spread.loc[larger_date].values, spread.loc[smaller_date].values)
                one_row_dict["spread_cos"] = (spread_cos + 1.0) / 2.0
                # ========================= 计算特征值 =========================
                for key in feat_larger_data.keys():
                    mean_1, mean_2, cos_sim = self.cal_pair_feat(feat_larger_data.get(key), feat_smaller_data.get(key))
                    one_row_dict[f"{key}_larger_mean"] = mean_1
                    one_row_dict[f"{key}_smaller_mean"] = mean_2
                    one_row_dict[f"{key}_cos_sim"] = cos_sim
                one_row_dict["grid_env"] = feat_larger_data['num'][0]
                one_row_dict["week_day"] = feat_larger_data['num'][1]
                one_row_dict["season"] = feat_larger_data['num'][2]
                one_row_dict["is_holiday"] = feat_larger_data['num'][3]
                one_row_dict["days"] = (larger_date-smaller_date).days
                res.append(one_row_dict)

        res = pd.DataFrame(res)
        return res


    def run(self, df, date_range):
        def filter_date(df, date_range):
            filtered_dfs = []
            for start, end in date_range:
                # 将字符串转为datetime.date对象
                start_date = datetime.strptime(start, '%Y-%m-%d').date()
                end_date = datetime.strptime(end, '%Y-%m-%d').date()
                temp_df = df.loc[start_date:end_date]
                filtered_dfs.append(temp_df)
            new_df = pd.concat(filtered_dfs)
            return new_df

        da, rt, spread, feat = self.transfer_data(df)
        if len(date_range) != 0:
            da = filter_date(da, date_range)
            rt = filter_date(rt, date_range)
            spread = filter_date(spread, date_range)
            feat = filter_date(feat, date_range)
        all_train_df = self.cal_all_train_df(spread, feat)

        return spread, all_train_df
