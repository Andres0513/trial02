import numpy as np
from scipy.interpolate import make_interp_spline

def extract_feat_to_dict(feat_df):
    feat_dict = {}
    for date in feat_df.index:
        feat_dict[date] = {
            'bid': feat_df.loc[date, [f'day_日前竞价空间{i}' for i in range(96)]].values.astype(np.float32),
            'load': feat_df.loc[date, [f'day_统调负荷预测{i}' for i in range(96)]].values.astype(np.float32),
            'imported': feat_df.loc[date, [f'day_外来电计划{i}' for i in range(96)]].values.astype(np.float32),
            'pv': feat_df.loc[date, [f'day_光伏出力预测{i}' for i in range(96)]].values.astype(np.float32),
            'wind_power': feat_df.loc[date, [f'day_风电出力预测{i}' for i in range(96)]].values.astype(np.float32),
            'fixed': feat_df.loc[date, [f'day_固定出力计划{i}' for i in range(96)]].values.astype(np.float32),
            'temp': feat_df.loc[date, [f'day_温度{i}' for i in range(24)]].values.astype(np.float32),
            'humidity': feat_df.loc[date, [f'day_湿度{i}' for i in range(24)]].values.astype(np.float32),
            'rain': feat_df.loc[date, [f'day_降雨{i}' for i in range(24)]].values.astype(np.float32),
            'irr': feat_df.loc[date, [f'day_辐照{i}' for i in range(24)]].values.astype(np.float32),
            'cloud': feat_df.loc[date, [f'day_云{i}' for i in range(24)]].values.astype(np.float32),
            'wind': feat_df.loc[date, [f'day_风速{i}' for i in range(24)]].values.astype(np.float32),
            'num': feat_df.loc[date, ['grid_env', '星期', '季度', '是否节假日']].values.astype(np.float32)
        }
        feat_dict[date] = {k: (v if k == 'num' else align_curve(k, v, 96))
                           for k, v in feat_dict[date].items()}
    return feat_dict

def map_feat_key(feat_dict, test_flag=False):
    # 定义映射关系（旧名字 -> 新名字）
    mapping = {
        'bid': 'curve_0',
        'load': 'curve_1',
        'imported': 'curve_2',
        'pv': 'curve_3',
        'wind_power': 'curve_4',
        'fixed': 'curve_5',
        'temp': 'curve_6',
        'humidity': 'curve_7',
        'rain': 'curve_8',
        'irr': 'curve_9',
        'cloud': 'curve_10',
        'wind': 'curve_11',
        'num': 'numerical_feat'
    }
    if test_flag:
        new_dict = {mapping.get(k, k)+'_2': v for k, v in feat_dict.items()}
        return new_dict
    else:
        new_dict = {mapping.get(k, k): v for k, v in feat_dict.items()}
        return new_dict

def align_curve(curve_name, curve_data, target_seq_len):
    if len(curve_data) == target_seq_len:
        return curve_data

    original_x = np.arange(len(curve_data))
    target_x = np.linspace(0, len(curve_data) - 1, target_seq_len)

    # 负荷数据线性插值，气象曲线使用三次样条插值保证平滑
    if (curve_name == 'temp' or curve_name == 'humidity' or curve_name == 'rain' or
            curve_name == 'irr' or curve_name == 'cloud' or curve_name == 'wind'):
        spl = make_interp_spline(original_x, curve_data, k=3)
        return spl(target_x).astype(np.float32)
    else:
        return np.interp(target_x, original_x, curve_data).astype(np.float32)