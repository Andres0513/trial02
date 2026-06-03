import pandas as pd
import numpy as np

from plot_figures import Plot_Figures
from enums import similarity_method
from data_loader import Data_Loader
from attention_model import *

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from tqdm import tqdm  # 用于显示进度条，如果没有请 pip install tqdm


SIMILARITY_METHOD = similarity_method.COSINE


if __name__ == '__main__':
    # 加载保存的数据集
    # df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
    df = pd.read_csv("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.csv", index_col=0)
    print(f"✅ 数据集已加载：uninted_df.pkl")

    dl = Data_Loader(SIMILARITY_METHOD)
    # 需要的数据日期范围
    date_range = [['2024-05-02', '2026-04-30']]
    da, rt, spread, feat, scores = dl.run(df, date_range, True)

    # scores.to_excel('cosine_scores.xlsx')
    # plot_spread_distribution(df)
    # plot_two_days_spread(spread, '2024-12-13', '2024-12-21')

    # 相似
    # date_str1 = '2024-12-13'
    # date_str2 = '2024-12-21'
    # date_str1 = '2024-06-03'
    # date_str2 = '2026-04-29'
    # date_str1 = '2025-06-08'
    # date_str2 = '2025-09-08'
    # date_str1 = '2026-01-10'
    # date_str2 = '2026-03-02'
    # date_str1 = '2025-12-20'
    # date_str2 = '2026-01-19'
    # 很不相似
    # date_str1 = '2024-09-22'
    # date_str2 = '2024-12-13'
    # date_str1 = '2024-06-01'
    # date_str2 = '2024-09-30'
    # cosine相似
    # date_str1 = '2024-12-02'
    # date_str2= '2024-12-17'
    # date_str1 = '2025-05-12'
    # date_str2 = '2026-04-19'
    # date_str1 = '2025-01-20'
    # date_str2 = '2026-04-26'
    # date_str1 = '2025-03-17'
    # date_str2 = '2026-03-10'
    # # cosine不相似
    # date_str1 = '2025-03-25'
    # date_str2 = '2026-03-10'
    # date_str1 = '2025-03-26'
    # date_str2 = '2026-03-10'
    # date_str1 = '2026-02-26'
    # date_str2 = '2026-04-09'
    # date_str1 = '2024-06-06'
    # date_str2 = '2025-09-04'
    # pf = Plot_Figures()
    # pf.plot_two_days_spread_and_feat(spread, feat, date_str1, date_str2)

    # 1. 设备配置 (针对 M1/M2/M3 芯片)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 已启用 Apple MPS 加速")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU 训练")

    # 2. 数据准备
    full_dataset = DatePairDataset(scores, feat, 96)  # 假设你已经加载了 df_scores 和 df_feat

    # 划分数据集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    # 创建 Loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 3. 模型初始化
    model = UltimateSiameseNet(num_numerical_features=4).to(device)
    # criterion = nn.MSELoss()  # 回归任务用 MSE，或者 BCEWithLogitsLoss (取决于标签范围)

    # 替换原来的 nn.MSELoss()
    def weighted_mse(pred, target):
        weight = torch.ones_like(target)
        weight[target > 0.7] = 2.0  # 高分更重要
        weight[target < 0.4] = 2.0  # 低分更重要
        weight[target > 0.8] = 4.0  # 高分更重要
        weight[target < 0.2] = 4.0  # 低分更重要
        weight[target > 0.9] = 8.0  # 高分更重要
        weight[target < 0.1] = 8.0  # 低分更重要
        return torch.mean(weight * (pred - target) ** 2)


    criterion = weighted_mse  # 直接赋值

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

    def move_batch_to_device(batch_data, device):
        """
        增强版：兼容 List, Tuple, Tensor
        """
        if isinstance(batch_data, torch.Tensor):
            # 如果是 Tensor，直接移动
            return batch_data.to(device)
        elif isinstance(batch_data, (list, tuple)):
            # 如果是 List 或 Tuple，递归处理里面的每一个元素
            # 注意：这里使用 type(batch_data) 来保持原来的类型 (List 还是 Tuple)
            return type(batch_data)(move_batch_to_device(item, device) for item in batch_data)
        else:
            # 兜底，防止其他类型出错
            raise TypeError(f"Unsupported data type: {type(batch_data)}")


    # 5. 开始训练
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # --- 训练阶段 ---
        # 【关键修改】这里必须用 3 个变量接收：input1_tuple, input2_tuple, labels
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch_input1, batch_input2, batch_labels in pbar:
            # 移动数据到设备
            batch_input1 = move_batch_to_device(batch_input1, device)
            batch_input2 = move_batch_to_device(batch_input2, device)
            batch_labels = batch_labels.to(device)

            # 前向传播
            outputs = model(batch_input1, batch_input2)

            # 计算损失
            loss = criterion(outputs, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_input1, batch_input2, batch_labels in val_loader:
                batch_input1 = move_batch_to_device(batch_input1, device)
                batch_input2 = move_batch_to_device(batch_input2, device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_input1, batch_input2)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Epoch {epoch + 1} 完成 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

    # ========================== 验证集结果验证与整理 ==========================
    model.eval()
    results = []
    val_indices = val_dataset.indices

    with torch.no_grad():
        for idx in tqdm(val_indices, desc="验证集预测"):
            # 从原始数据集获取数据
            (input1, input2, label) = full_dataset[idx]
            # 获取原始日期对
            date1, date2 = full_dataset.pairs[idx]

            # 调整数据形状并移动到设备（增加batch维度）
            input1 = move_batch_to_device(tuple(t.unsqueeze(0) for t in input1), device)
            input2 = move_batch_to_device(tuple(t.unsqueeze(0) for t in input2), device)

            # 模型预测
            pred = model(input1, input2)

            # 收集结果
            results.append({
                'date1': date1,
                'date2': date2,
                '原始score': label.item(),
                '预测值': pred.item()
            })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    # 可选：保存结果到Excel
    results_df.to_excel('验证集预测结果.xlsx', index=False)
    print("✅ 验证集结果已整理完成，保存为 验证集预测结果.xlsx")
    print("\n验证集结果预览：")
    print(results_df.head())

    # ========================== 训练完以后，把模型转成 ONNX ==========================
    model.eval()  # 👉 导出前必须切换到评估模式，固定 Dropout 和 BatchNorm
    batch_size = 64
    seq_len = 96
    num_features = 4

    # 生成 dummy data
    dummy_curves = [torch.randn(batch_size, seq_len).to(device) for _ in range(12)]
    dummy_nums = torch.randn(batch_size, num_features).to(device)
    dummy_curves_2 = [torch.randn(batch_size, seq_len).to(device) for _ in range(12)]
    dummy_nums_2 = torch.randn(batch_size, num_features).to(device)

    # 组织成模型 forward 期望的格式：( (curve1, curve2, ..., num), (curve1, ..., num) )
    # 注意：这里我们使用元组嵌套，更符合 Python 的语义
    input1 = (*dummy_curves, dummy_nums)
    input2 = (*dummy_curves_2, dummy_nums_2)

    # 3. 导出配置
    input_names = [f"curve_{i}" for i in range(12)] + ["numerical_feat"] + \
                  [f"curve_{i}_2" for i in range(12)] + ["numerical_feat_2"]

    output_names = ["similarity_score"]

    # 动态轴：假设第 0 维是 batch
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size"}
    dynamic_axes[output_names[0]] = {0: "batch_size"}

    # 4. 执行导出 (关键点：dynamo=True)
    torch.onnx.export(
        model,
        (input1, input2),  # 直接传入两个元组，新后端能理解这种结构
        "ultimate_siamese_net.onnx",
        opset_version=16,
        do_constant_folding=True,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("✅ 模型已成功导出为 ultimate_siamese_net.onnx")