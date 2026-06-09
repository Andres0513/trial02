import pandas as pd


def sort_df(df, sort_key):
    df = df[df['target_date'] != df['reference_date']]
    sorted_df = df.sort_values(by=['target_date', sort_key], ascending=[True, False])

    return sorted_df

def cal_spread(sorted_df, spread, top_n):
    # 最终final_df，列分别为 target_date, reference_date, 时刻
    # 1. 取 top_n
    top_n_ref = (
        sorted_df.sort_values(['target_date', 'pred_y'], ascending=[True, False])
        .groupby('target_date', group_keys=False)
        .head(top_n)
    )

    # 2. 统一日期为 date 对象（去掉时间）
    top_n_pairs = top_n_ref[['target_date', 'reference_date']].copy()
    top_n_pairs['reference_date'] = pd.to_datetime(top_n_pairs['reference_date']).dt.date
    top_n_pairs['target_date'] = pd.to_datetime(top_n_pairs['target_date']).dt.date

    # 3. 复制 spread_df，防止影响原始数据， 并重命名为 reference_date
    spread_df = spread.copy()
    spread_df.rename(columns={
        '日期': 'date'
    }, inplace=True)
    # spread_df.rename(columns={'index': 'date'}, inplace=True)  # 改名叫 date
    # spread_df['date'] = pd.to_datetime(spread_df['日期']).dt.date  # 统一为 date

    # 3. 按 target_date 循环匹配
    result_dfs = []
    for target_date, group in top_n_pairs.groupby('target_date'):
        ref_dates = group['reference_date'].tolist()

        # 用 date 列匹配
        sub_spread = spread_df[spread_df['date'].isin(ref_dates)].copy()
        if sub_spread.empty:
            print(f"警告：{target_date} 无有效 reference_date")
            continue

        sub_spread['target_date'] = target_date
        # --- 把 target_date 挪到第一列 ---
        sub_spread.insert(0, 'target_date', sub_spread.pop('target_date'))
        sub_spread.columns = [f"{col}" for col in sub_spread.columns]
        result_dfs.append(sub_spread)

    # 4. 拼接
    final_df = pd.concat(result_dfs, axis=0)

    # 5. 规范化，重命名 date 列为 reference_date。最终final_df，列分别为 target_date, reference_date, 时刻
    final_df.rename(columns={'date': 'reference_date'}, inplace=True)
    return final_df