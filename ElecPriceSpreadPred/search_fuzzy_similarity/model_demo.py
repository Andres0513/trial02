import pandas as pd
from enums import similarity_method
from fuzzy_similarity_data import Fuzzy_Similarity_Data
from datetime import date, timedelta, time
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def split_train_and_test(all_df):
    val_split_date = date(2026,4,28)
    test_split_date = date(2026,5,1)
    train_df = all_df[all_df['target_date'] < val_split_date]
    val_df = all_df[(all_df['target_date'] >= val_split_date) & (all_df['target_date'] < test_split_date)]
    test_df = all_df[all_df['target_date'] >= test_split_date]
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df

def validate_model(model, x, true_y):
    pred_y = model.predict(x)
    # 计算指标
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    return pred_y, rmse

def sort_df(df, sort_key):
    df = df[df['target_date'] != df['reference_date']]
    sorted_df = df.sort_values(by=['target_date', sort_key], ascending=[True, False])

    return sorted_df

def cal_spread(sorted_df, spread, top_n):
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

    # 3. 索引变列 → 列名叫 index → 重命名为 date（关键）
    spread_df = spread.reset_index()   # 索引变成列：列名 = index
    spread_df.rename(columns={
        '日期': 'reference_date'
    }, inplace=True)
    # spread_df.rename(columns={'index': 'date'}, inplace=True)  # 改名叫 date
    # spread_df['date'] = pd.to_datetime(spread_df['日期']).dt.date  # 统一为 date

    # 4. 按 target_date 循环匹配
    result_dfs = []
    for target_date, group in top_n_pairs.groupby('target_date'):
        ref_dates = group['reference_date'].tolist()

        # 用 date 列匹配
        sub_spread = spread_df[spread_df['reference_date'].isin(ref_dates)].copy()
        if sub_spread.empty:
            print(f"警告：{target_date} 无有效 reference_date")
            continue

        sub_spread['target_date'] = target_date
        # --- 把 target_date 挪到第一列 ---
        sub_spread.insert(0, 'target_date', sub_spread.pop('target_date'))
        sub_spread.columns = [f"{col}" for col in sub_spread.columns]
        result_dfs.append(sub_spread)

    # 5. 拼接
    final_df = pd.concat(result_dfs, axis=0)
    return final_df

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
    train_df, val_df, test_df = split_train_and_test(all_df)
    #  ================= 数据分布 =================
    # 5% 分位数
    q5 = np.percentile(train_df['spread_cos'], 5)
    # 95% 分位数
    q95 = np.percentile(train_df['spread_cos'], 95)

    print(f"5% 分位数: {q5:.5f}")
    print(f"95% 分位数: {q95:.5f}")

    # ================= 重采样 =================
    # 1. 选出要增强的样本
    high_score_df = train_df[train_df['spread_cos'] > 0.75].copy()
    low_score_df = train_df[train_df['spread_cos'] < 0.25].copy()
    high_score_df_2 = train_df[train_df['spread_cos'] > 0.9].copy()
    low_score_df_2 = train_df[train_df['spread_cos'] < 0.1].copy()

    # 2. 每个样本复制 20 次（最安全、最快写法）
    repeat_times = 20
    oversample_df_1 = pd.concat([high_score_df] * repeat_times, ignore_index=True)
    oversample_df_2 = pd.concat([low_score_df] * repeat_times, ignore_index=True)
    oversample_df_3 = pd.concat([high_score_df_2] * repeat_times, ignore_index=True)
    oversample_df_4 = pd.concat([low_score_df_2] * repeat_times, ignore_index=True)

    # 3. 追加回训练集
    train_df = pd.concat([train_df, oversample_df_1], ignore_index=True)
    train_df = pd.concat([train_df, oversample_df_2], ignore_index=True)
    train_df = pd.concat([train_df, oversample_df_3], ignore_index=True)
    train_df = pd.concat([train_df, oversample_df_4], ignore_index=True)

    # ================= 对日期进行预处理 =================
    train_df_backup = train_df.copy()
    val_df_backup = val_df.copy()
    test_df_backup = test_df.copy()
    train_df = train_df.drop(columns=['target_date', 'reference_date'])
    val_df = val_df.drop(columns=['target_date', 'reference_date'])
    test_df = test_df.drop(columns=['target_date', 'reference_date'])

    # 枚举特征：周几、季度、电网工况 → 转成 category 类型
    categorical_cols = []
    for col in train_df.columns:
        if 'grid_env' in col or 'week_day' in col or 'season' in col or 'is_holiday' in col:
            categorical_cols.append(col)

    # 转 category（XGBoost 支持直接训练）
    train_df[categorical_cols] = train_df[categorical_cols].astype('int').astype('category')
    val_df[categorical_cols] = val_df[categorical_cols].astype('int').astype('category')
    test_df[categorical_cols] = test_df[categorical_cols].astype('int').astype('category')

    # ================= 划分输入输出 =================
    train_x = train_df.iloc[:, 0:]
    train_y = train_df.iloc[:, 0]
    val_x = val_df.iloc[:, 0:]
    val_y = val_df.iloc[:, 0]
    test_x = test_df.iloc[:, 0:]
    test_y = test_df.iloc[:, 0]

    # 模型
    model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=1,
        min_child_weight=2,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        objective="reg:squarederror",  # 回归损失
        eval_metric="rmse"  # 回归常用评价：rmse / mae
    )

    # 训练
    model.fit(train_x, train_y)
    # 训练集
    pred_y_train, rmse_train = validate_model(model, train_x, train_y)
    # 验证集
    pred_y_val, rmse_val = validate_model(model, val_x, val_y)
    # 测试集
    pred_y_test, rmse_test = validate_model(model, test_x, test_y)

    val_res = pd.concat([val_df_backup, pd.DataFrame(pred_y_val, columns=['pred_y'])], axis=1)
    # 把 pred_y 移到第一列
    cols = ['pred_y'] + [c for c in val_res.columns if c != 'pred_y']
    val_res = val_res[cols]

    test_res = pd.concat([test_df_backup, pd.DataFrame(pred_y_test, columns=['pred_y'])], axis=1)
    # 把 pred_y 移到第一列
    cols = ['pred_y'] + [c for c in test_res.columns if c != 'pred_y']
    test_res = test_res[cols]

    val_res = sort_df(val_res, 'pred_y')
    final_test_res = cal_spread(val_res, spread, 10)

    # ================= 计算宽表 =================
    mean_spread_df = final_test_res.groupby("target_date").mean(numeric_only=True).reset_index()
    mean_spread_df.insert(1, 'reference_date', 'pred_spread')
    # mean_spread_df.insert(2, 'pred_y', 'pred_spread')
    # mean_spread_df.insert(3, 'spread_cos', 'pred_spread')
    wide_res = pd.concat([final_test_res, mean_spread_df], axis=0)
    wide_res = wide_res.merge(val_res[['target_date', 'reference_date', 'pred_y', 'spread_cos']], on=['target_date', 'reference_date'], how='left')
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
    new_spread_df = spread[spread.index.isin(target_dates)].copy()
    new_spread_df = new_spread_df.reset_index(names='target_date')
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
            else:
                consistency = np.nan
        else:
            consistency = np.nan

        consistency_data.append({
            'target_date': target_date,
            'sign_consistency': consistency
        })

    # 转成DataFrame，方便merge
    consistency_df = pd.DataFrame(consistency_data)

    # 3. 合并回wide_res，只在reference_date='raw_spread'的行填充
    wide_res = wide_res.merge(consistency_df, on='target_date', how='left')
    wide_res.loc[wide_res['reference_date'] != 'raw_spread', 'sign_consistency'] = np.nan

    # 4. 把sign_consistency插入到第5列（索引4）
    cols = wide_res.columns.tolist()
    cols.insert(4, cols.pop(cols.index('sign_consistency')))
    wide_res = wide_res[cols]

    a = 0
