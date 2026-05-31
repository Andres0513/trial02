import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


# ==========================================
# 1. 数据预处理与 Dataset (无缝集成插值对齐)
# ==========================================
class DatePairDataset(Dataset):
    """
    支持异构时序数据的 Dataset
    自动在读取时将不同长度的曲线对齐到目标长度
    """

    def __init__(self, df_scores, df_feat, target_seq_len=96):
        self.pairs = []
        self.scores = []
        self.target_seq_len = target_seq_len

        # 提前将 DataFrame 转为字典，加快后续读取速度
        self.feat_dict = {}
        for date in df_feat.index:
            self.feat_dict[date] = {
                'bid': df_feat.loc[date, [f'day_日前竞价空间{i}' for i in range(96)]].values.astype(np.float32),
                'load': df_feat.loc[date, [f'day_统调负荷预测{i}' for i in range(96)]].values.astype(np.float32),
                'imported': df_feat.loc[date, [f'day_外来电计划{i}' for i in range(96)]].values.astype(np.float32),
                'pv': df_feat.loc[date, [f'day_光伏出力预测{i}' for i in range(96)]].values.astype(np.float32),
                'wind_power': df_feat.loc[date, [f'day_风电出力预测{i}' for i in range(96)]].values.astype(np.float32),
                'fixed': df_feat.loc[date, [f'day_固定出力计划{i}' for i in range(96)]].values.astype(np.float32),
                'temp' : df_feat.loc[date, [f'day_温度{i}' for i in range(24)]].values.astype(np.float32),
                'humidity': df_feat.loc[date, [f'day_湿度{i}' for i in range(24)]].values.astype(np.float32),
                'rain': df_feat.loc[date, [f'day_降雨{i}' for i in range(24)]].values.astype(np.float32),
                'irr': df_feat.loc[date, [f'day_辐照{i}' for i in range(24)]].values.astype(np.float32),
                'cloud': df_feat.loc[date, [f'day_云{i}' for i in range(24)]].values.astype(np.float32),
                'wind': df_feat.loc[date, [f'day_风速{i}' for i in range(24)]].values.astype(np.float32),
                'num': df_feat.loc[date, ['grid_env', '星期', '季度', '是否节假日']].values.astype(np.float32)
            }

        for _, row in df_scores.iterrows():
            self.pairs.append((row['date1'], row['date2']))
            self.scores.append(row['similarity_score'])

    def _align_curve(self, curve_name, curve_data):
        """
        根据曲线类型智能选择插值算法并对齐到目标长度
        """
        if len(curve_data) == self.target_seq_len:
            return curve_data

        original_x = np.arange(len(curve_data))
        target_x = np.linspace(0, len(curve_data) - 1, self.target_seq_len)

        # 负荷数据线性插值，气象曲线使用三次样条插值保证平滑
        if (curve_name == 'temp' or curve_name == 'humidity' or curve_name == 'rain' or
              curve_name == 'irr' or curve_name == 'cloud' or curve_name == 'wind'):
            spl = make_interp_spline(original_x, curve_data, k=3)
            return spl(target_x).astype(np.float32)
        else:
            return np.interp(target_x, original_x, curve_data).astype(np.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        d1, d2 = self.pairs[idx]
        score = self.scores[idx]

        # 提取日期1的数据并自动对齐
        feat1_raw = self.feat_dict[d1]
        feat1 = {k: self._align_curve(k, v) for k, v in feat1_raw.items() if k != 'num'}
        feat1['num'] = feat1_raw['num']

        # 提取日期2的数据并自动对齐
        feat2_raw = self.feat_dict[d2]
        feat2 = {k: self._align_curve(k, v) for k, v in feat2_raw.items() if k != 'num'}
        feat2['num'] = feat2_raw['num']

        return (
            (torch.from_numpy(feat1['bid']), torch.from_numpy(feat1['load']),
             torch.from_numpy(feat1['imported']), torch.from_numpy(feat1['pv']),
             torch.from_numpy(feat1['wind_power']), torch.from_numpy(feat1['fixed']),
             torch.from_numpy(feat1['temp']), torch.from_numpy(feat1['humidity']),
             torch.from_numpy(feat1['rain']), torch.from_numpy(feat1['irr']),
             torch.from_numpy(feat1['cloud']), torch.from_numpy(feat1['wind']),
             torch.from_numpy(feat1['num'])),
            (torch.from_numpy(feat2['bid']), torch.from_numpy(feat2['load']),
             torch.from_numpy(feat2['imported']), torch.from_numpy(feat2['pv']),
             torch.from_numpy(feat2['wind_power']), torch.from_numpy(feat2['fixed']),
             torch.from_numpy(feat2['temp']), torch.from_numpy(feat2['humidity']),
             torch.from_numpy(feat2['rain']), torch.from_numpy(feat2['irr']),
             torch.from_numpy(feat2['cloud']), torch.from_numpy(feat2['wind']),
             torch.from_numpy(feat2['num'])),
            torch.tensor(score, dtype=torch.float32)
        )


# ==========================================
# 2. 模型核心组件 (自注意力 & 动态交叉融合)
# ==========================================
class SelfAttentionBlock(nn.Module):
    """单曲线自注意力模块：捕捉单条曲线的关键形状特征"""

    def __init__(self, seq_len=96, hidden_dim=32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, seq_len),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attention(x)
        weighted_x = x * weights
        return weighted_x, weights


class DynamicCrossFusion(nn.Module):
    """
    动态多曲线交叉融合模块
    支持任意数量的曲线输入，自动完成所有曲线间的交叉注意力计算
    """

    def __init__(self, feature_dim=32, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, all_curves_feat):
        # Q, K, V 都是同一个张量：让所有的曲线互相“看”对方
        fused_out, _ = self.cross_attn(query=all_curves_feat, key=all_curves_feat, value=all_curves_feat)
        # 残差连接 + LayerNorm，防止梯度消失
        out = self.norm(all_curves_feat + fused_out)
        return out


# ==========================================
# 3. 终极孪生网络主模型
# ==========================================
class UltimateSiameseNet(nn.Module):
    def __init__(self, num_numerical_features=4):
        super().__init__()

        # 1. 为每一条曲线建立独立的自注意力提取器
        self.curve_names = ['bid', 'load', 'imported', 'pv', 'wind_power', 'fixed',
                            'temp', 'humidity', 'rain', 'irr', 'cloud', 'wind']
        self.self_attns = nn.ModuleDict({
            name: SelfAttentionBlock(seq_len=96, hidden_dim=32) for name in self.curve_names
        })
        self.curve_fcs = nn.ModuleDict({
            name: nn.Linear(96, 32) for name in self.curve_names
        })

        # 2. 【核心】通用的多曲线交叉融合模块
        self.dynamic_fusion = DynamicCrossFusion(feature_dim=32, num_heads=4)

        # 3. 数值特征处理
        self.num_fc = nn.Sequential(nn.Linear(num_numerical_features, 16), nn.ReLU())

        # 4. 最终回归层
        # 维度计算：N条曲线自身特征(32*N) + 数值特征(16) + N条交叉后特征(32*N)
        total_dim = len(self.curve_names) * 32 + 16 + len(self.curve_names) * 32
        self.regressor = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def extract_features(self, curves_tuple, num_data):
        """
        提取单个日期的综合特征向量
        """
        batch_size = num_data.size(0)
        curve_feat_list = []

        # A. 遍历所有曲线，提取自身特征
        for i, name in enumerate(self.curve_names):
            w, _ = self.self_attns[name](curves_tuple[i])
            feat = F.relu(self.curve_fcs[name](w))  # (batch, 32)
            curve_feat_list.append(feat)

        # B. 【关键步骤】把所有曲线特征堆叠成一个三维张量 (Batch, N, 32)
        all_curves_stacked = torch.stack(curve_feat_list, dim=1)

        # C. 一键完成所有曲线间的交叉融合
        fused_curves = self.dynamic_fusion(all_curves_stacked)  # (Batch, N, 32)

        # D. 展平并拼接数值特征
        flat_orig_feats = all_curves_stacked.view(batch_size, -1)  # (Batch, N*32)
        flat_fused_feats = fused_curves.view(batch_size, -1)  # (Batch, N*32)
        num_feat = self.num_fc(num_data)  # (Batch, 16)

        final_feat = torch.cat([flat_orig_feats, flat_fused_feats, num_feat], dim=1)
        return final_feat

    def forward(self, input1, input2):
        # input1, input2 都是包含 (load, wind, temp, irr, num) 的元组
        curves1, num1 = input1[:-1], input1[-1]
        curves2, num2 = input2[:-1], input2[-1]

        feat1 = self.extract_features(curves1, num1)
        feat2 = self.extract_features(curves2, num2)

        # 计算两个综合特征向量的绝对差值 (L1距离)
        diff = torch.abs(feat1 - feat2)

        # 预测得分
        score = self.regressor(diff)
        return score.squeeze()


# ==========================================
# 4. 训练流程示例
# ==========================================
if __name__ == '__main__':
    # 假设你已经准备好了 df_scores 和 df_feat
    # full_dataset = DatePairDataset(df_scores, df_feat)
    # train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltimateSiameseNet(num_numerical_features=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    print("模型结构如下：")
    print(model)
    print(f"\n当前设备: {device}")
    print("准备就绪，可以开始你的训练循环啦！")