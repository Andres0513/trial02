import onnxruntime as ort
import pandas as pd
import numpy as np
from data_loader import Data_Loader
from enums import similarity_method
import utils

SIMILARITY_METHOD = similarity_method.COSINE

session = ort.InferenceSession("ultimate_siamese_net_epoch300_非加权误差.onnx")
print("✅ 模型已成功加载 ultimate_siamese_net.onnx")
# df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
df = pd.read_csv("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.csv", index_col=0)
# 需要的数据日期范围
date_range = [['2026-05-01', '2026-05-14']]
dl = Data_Loader(SIMILARITY_METHOD)
da_test, rt_test, spread_test, feat_test, _ = dl.run(df, date_range, True)
feat_test_dict = utils.extract_feat_to_dict(feat_test)


date_range = [['2024-05-02', '2026-04-30']]
da_ref, rt_ref, spread_ref, feat_ref, _ = dl.run(df, date_range, True)
feat_ref_dict = utils.extract_feat_to_dict(feat_ref)

# 初始化结果空 DataFrame（三列：测试日期、参考日期、模型输出结果）
fitted_similarity_df = pd.DataFrame(columns=['date_test', 'date_ref', 'fitted_similarity'])
for date_test, data_test in feat_test.iterrows():
    for date_ref, data_ref in feat_ref.iterrows():
        feat_1 = feat_test_dict.get(date_test)
        feat_1 = utils.map_feat_key(feat_1, True)
        feat_2 = feat_ref_dict.get(date_ref)
        feat_2 = utils.map_feat_key(feat_2, False)
        inputs_dict = feat_1.copy()
        inputs_dict.update(feat_2)
        for key in ['numerical_feat', 'numerical_feat_2'] + [f"curve_{i}" for i in range(12)] + [f"curve_{i}_2" for i in range(12)]:
            inputs_dict[key] = np.expand_dims(inputs_dict[key], axis=0)
        results = session.run(None, inputs_dict)
        # 把当前这一组 test/ref/结果 追加到结果表
        new_row = {
            'date_test': date_test,
            'date_ref': date_ref,
            'fitted_similarity': float(results[0])  # 如果是数组会自动存为列表/数值
        }
        # 追加一行
        fitted_similarity_df = pd.concat([fitted_similarity_df, pd.DataFrame([new_row])], ignore_index=True)
        a = 0

fitted_similarity_df['fitted_similarity'] = pd.to_numeric(fitted_similarity_df['fitted_similarity'], errors='coerce')
# 按 date_test 分组，并在每个组内对 result 列取最大的 n 个值
top_n_df = fitted_similarity_df.groupby('date_test').apply(
    lambda x: x.nlargest(5, 'fitted_similarity')
).reset_index(drop=False)

# 为了美观，按日期和分数降序排列
top_n_df = top_n_df.sort_values(by=['date_test', 'fitted_similarity'], ascending=[True, False]).reset_index(drop=True)

#  把 spread_ref 的日期索引转为列，方便合并
spread_ref_reset = spread_ref.reset_index(names="date_ref")
# 把 top_10_df 和 spread_ref 合并，拿到每个 date_ref 对应的 spread 数据
wide_df = pd.merge(
    top_n_df[["date_test", "date_ref", "fitted_similarity"]],
    spread_ref_reset,
    on="date_ref",
    how="left"
)

#  按 date_test 分组，对所有时间列取均值（避免循环）
# 这里只对数值列做平均，非数值列会被自动忽略
mean_spread_df = wide_df.groupby("date_test").mean(numeric_only=True)

# 调整成和 spread_test 完全一样的维度/顺序
# 按 spread_test 的索引排序，列也对齐
mean_spread_df = mean_spread_df.reindex(
    index=spread_test.index,
    columns=spread_test.columns
)

def calc_sign_consistency(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    same_sign = np.sign(df1.values) == np.sign(df2.values)
    daily_ratio = same_sign.mean(axis=1)
    res_df = pd.DataFrame({"consist_ratio": daily_ratio}, index=df1.index)
    return res_df

# 计算正负一致性
sign_consistency = calc_sign_consistency(mean_spread_df, spread_test)
avg = sign_consistency['consist_ratio'].mean()

wide_df = pd.merge(
    wide_df,
    sign_consistency.reset_index(names='date_test'),
    on=['date_test'],
    how='left'
)

wide_df['avg'] = avg

# ========== 把均值的列往前移动，移动到第四列 ==========
cols = wide_df.columns.to_list()
cols.remove('avg')
cols.remove('consist_ratio')
cols.insert(3, 'avg')
cols.insert(4, 'consist_ratio')
wide_df = wide_df[cols]

# ========== 把平均值也插入到宽表里去 ============
mean_spread_df['date_ref'] = pd.NA
mean_spread_df['fitted_similarity'] = pd.NA
mean_spread_df['consist_ratio'] = pd.NA
mean_spread_df['avg'] = avg
mean_spread_df = mean_spread_df.reset_index(names='date_test')
cols = mean_spread_df.columns.to_list()
cols.remove('date_ref')
cols.remove('fitted_similarity')
cols.remove('avg')
cols.remove('consist_ratio')
cols.insert(1, 'date_ref')
cols.insert(2, 'fitted_similarity')
cols.insert(3, 'avg')
cols.insert(4, 'consist_ratio')
mean_spread_df = mean_spread_df[cols]

# 所有的信息拼接完成，并排序好
wide_df = pd.concat([wide_df, mean_spread_df], ignore_index=True)
wide_df = wide_df.sort_values(by=['date_test', 'fitted_similarity'], ascending=[True, False]).reset_index(drop=True)

wide_df.to_excel("result_wide_df.xlsx")
print(f"平均准确率{avg:.6f}")
a = 0