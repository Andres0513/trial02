import onnxruntime as ort
import pandas as pd
import numpy as np
from data_loader import Data_Loader
from enums import similarity_method
import utils

SIMILARITY_METHOD = similarity_method.COSINE

session = ort.InferenceSession("ultimate_siamese_net.onnx")
print("✅ 模型已成功加载 ultimate_siamese_net.onnx")
df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
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
for date_test, data_test in feat_ref.iterrows():
    for date_ref, data_ref in feat_ref.iterrows():
        feat_1 = feat_ref_dict.get(date_test)
        feat_1 = utils.map_feat_key(feat_1, True)
        feat_2 = feat_ref_dict.get(date_ref)
        feat_2 = utils.map_feat_key(feat_2, False)
        inputs_dict = feat_1.copy()
        inputs_dict.update(feat_2)
        for key in ['numerical_feat', 'numerical_feat_2'] + [f"curve_{i}" for i in range(12)] + [f"curve_{i}_2" for i in range(12)]:
            inputs_dict[key] = np.expand_dims(inputs_dict[key], axis=0)

        print(inputs_dict)
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

result_df.to_excel('result_df.xlsx')
a = 0