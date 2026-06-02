import onnxruntime as ort
import pandas as pd
import numpy as np
from data_loader import Data_Loader
from enums import similarity_method
import utils

SIMILARITY_METHOD = similarity_method.COSINE

session = ort.InferenceSession("ultimate_siamese_net.onnx")
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
result_df = pd.DataFrame(columns=['date_test', 'date_ref', 'result'])
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
            'result': float(results[0])  # 如果是数组会自动存为列表/数值
        }
        # 追加一行
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
        a = 0

result_df['result'] = pd.to_numeric(result_df['result'], errors='coerce')
# 1. 按 date_test 分组，并在每个组内对 result 列取最大的 10 个值
top_10_df = result_df.groupby('date_test').apply(
    lambda x: x.nlargest(5, 'result')
).reset_index(drop=False)

# 2. (可选) 为了美观，可以按日期和分数降序排列
top_10_df = top_10_df.sort_values(by=['date_test', 'result'], ascending=[True, False]).reset_index(drop=True)


# 1. 把 spread_ref 的日期索引转为列，方便合并
spread_ref_reset = spread_ref.reset_index(names="date_ref")
# 2. 把 top_10_df 和 spread_ref 合并，拿到每个 date_ref 对应的 spread 数据
merged = pd.merge(
    top_10_df[["date_test", "date_ref"]],
    spread_ref_reset,
    on="date_ref",
    how="left"
)

# 3. 按 date_test 分组，对所有时间列取均值（避免循环）
# 这里只对数值列做平均，非数值列会被自动忽略
mean_spread_df = merged.groupby("date_test").mean(numeric_only=True)

# 4. 调整成和 spread_test 完全一样的维度/顺序
# 按 spread_test 的索引排序，列也对齐
mean_spread_df = mean_spread_df.reindex(
    index=spread_test.index,
    columns=spread_test.columns
)

result_df.to_excel('result_df.xlsx')
a = 0