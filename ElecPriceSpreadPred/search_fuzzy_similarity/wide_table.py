import pandas as pd
import numpy as np
from datetime import time

class Wide_Table:
    def __init__(self):
        return

    def run(self, final_res, spread, val_res):
        # ================= 计算宽表 =================
        mean_spread_df = final_res.groupby("target_date").mean(numeric_only=True).reset_index()
        mean_spread_df.insert(1, 'reference_date', 'pred_spread')
        # mean_spread_df.insert(2, 'pred_y', 'pred_spread')
        # mean_spread_df.insert(3, 'spread_cos', 'pred_spread')
        wide_res = pd.concat([final_res, mean_spread_df], axis=0)
        wide_res = wide_res.merge(val_res[['target_date', 'reference_date', 'pred_y', 'spread_cos']],
                                  on=['target_date', 'reference_date'], how='left')
        # 调整列顺序：先获取原有列名，再重新排列
        # 1. 提取除了pred_y、spread_cos之外的所有列
        other_cols = [col for col in wide_res.columns if col not in ['pred_y', 'spread_cos']]
        # 2. 重新构造列顺序：前两列 + pred_y + spread_cos + 剩余列
        new_col_order = other_cols[:2] + ['pred_y', 'spread_cos'] + other_cols[2:]
        # 3. 按新顺序重新排列DataFrame
        wide_res = wide_res[new_col_order]

        # 取原始结果
        # 1. 提取mean_spread_df里的target_date列表
        target_dates = mean_spread_df['target_date'].unique()
        # 2. 在spread中筛选这些日期对应的行，生成新df
        new_spread_df = spread[spread['日期'].isin(target_dates)].copy()
        new_spread_df.rename(columns={'日期': 'target_date'}, inplace=True)
        new_spread_df.insert(1, 'reference_date', 'raw_spread')
        new_spread_df.insert(2, 'pred_y', 'raw_spread')
        new_spread_df.insert(3, 'spread_cos', 'raw_spread')
        # 把时刻列转成str
        new_spread_df.columns = [
            col.strftime("%H:%M:%S") if isinstance(col, time) else str(col)
            for col in new_spread_df.columns
        ]
        wide_res = pd.concat([wide_res, new_spread_df], axis=0, ignore_index=True)

        # =================== 计算一致性 =====================
        # 1. 先提取所有时刻列（从00:00:00开始的列）
        time_cols = [col for col in wide_res.columns if col.startswith(('00:', '01:', '02:', '03:', '04:', '05:', '06:',
                                                                        '07:', '08:', '09:', '10:', '11:', '12:', '13:',
                                                                        '14:', '15:', '16:', '17:', '18:', '19:', '20:',
                                                                        '21:', '22:', '23:'))]

        # 2. 按target_date分组，计算正负一致性
        consistency_data = []
        for target_date, group in wide_res.groupby('target_date'):
            # 找到pred_spread和raw_spread的行
            pred_row = group[group['reference_date'] == 'pred_spread']
            raw_row = group[group['reference_date'] == 'raw_spread']

            if len(pred_row) == 1 and len(raw_row) == 1:
                # 提取对应时刻的值
                pred_values = pred_row[time_cols].iloc[0].values
                raw_values = raw_row[time_cols].iloc[0].values

                # 过滤掉NaN的时刻
                mask = ~np.isnan(pred_values) & ~np.isnan(raw_values)
                pred_values = pred_values[mask]
                raw_values = raw_values[mask]

                if len(pred_values) > 0:
                    # 计算符号是否一致（>0为1，<0为-1，符号相乘>0即同号）
                    sign_match = (np.sign(pred_values) * np.sign(raw_values)) > 0
                    consistency = sign_match.mean()  # 同号比例，0~1
                    # True乘1，False乘-1，再乘以raw_values绝对值
                    weighted = np.where(sign_match, 1, -1) * np.abs(raw_values)
                    algo_spread = weighted.mean()
                    algo_spread_ratio = weighted.mean() / (np.abs(raw_values).mean()+1e-6)
                else:
                    consistency = np.nan
                    algo_spread = np.nan
            else:
                consistency = np.nan
                algo_spread = np.nan

            consistency_data.append({
                'target_date': target_date,
                'sign_consistency': consistency,
                'algo_spread' : algo_spread,
                'algo_spread_ratio': algo_spread_ratio
            })

        # 转成DataFrame，方便merge
        consistency_df = pd.DataFrame(consistency_data)

        # 3. 合并回wide_res，只在reference_date='pred_spread'的行填充
        wide_res = wide_res.merge(consistency_df, on='target_date', how='left')
        wide_res.loc[wide_res['reference_date'] != 'pred_spread', 'sign_consistency'] = np.nan
        wide_res.loc[wide_res['reference_date'] != 'pred_spread', 'algo_spread'] = np.nan
        wide_res.loc[wide_res['reference_date'] != 'pred_spread', 'algo_spread_ratio'] = np.nan

        # 4. 对‘pred_spread’的行填充nan
        wide_res.loc[wide_res['reference_date'] == 'pred_spread', ['pred_y', 'spread_cos']] = 'pred_spread'

        # 5. 把sign_consistency插入到第5列（索引4）
        cols = wide_res.columns.tolist()
        cols.insert(4, cols.pop(cols.index('sign_consistency')))
        cols.insert(5, cols.pop(cols.index('algo_spread')))
        cols.insert(6, cols.pop(cols.index('algo_spread_ratio')))
        wide_res = wide_res[cols]

        return wide_res