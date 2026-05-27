import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# 👇 解决中文乱码 + 负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class Plot_Figures:
    def plot_two_days_spread(self, spread_df, date_str1, date_str2):
        # 把字符串日期转成 date 对象（匹配你的索引）
        date1 = datetime.strptime(date_str1, '%Y-%m-%d').date()
        date2 = datetime.strptime(date_str2, '%Y-%m-%d').date()

        # 取出两天数据
        day1 = spread_df.loc[date1]
        day2 = spread_df.loc[date2]

        # ✅ 修复：把 time 对象转成字符串 00:00, 00:30...
        times = [t.strftime("%H:%M") for t in spread_df.columns]

        # 画图
        plt.figure(figsize=(14, 6))
        plt.plot(times, day1.values, marker='o', linewidth=2, label=date_str1)
        plt.plot(times, day2.values, marker='s', linewidth=2, label=date_str2)

        plt.title("两日价差曲线对比", fontsize=14)
        plt.xlabel("时间")
        plt.ylabel("价差")
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_spread_distribution(self,df):
        # 取出价差列，自动忽略 NaN
        spread_data = df['价差（实时-日前）'].dropna()
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        q_list = spread_data.quantile(quantiles)
        print("===== 价差（实时-日前） 分位数 =====")
        for q, val in q_list.items():
            print(f"{int(q * 100)} 分位数: {val:.4f}")

        plt.figure(figsize=(10, 6))

        # 1. 直方图 + 核密度曲线
        sns.histplot(spread_data, kde=True, bins=30, color='skyblue', edgecolor='black')

        plt.title("价差列的概率分布", fontsize=14)
        plt.xlabel("价差（实时-日前）", fontsize=12)
        plt.ylabel("频数 / 密度", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_two_days_spread_and_feat(self, spread, feat, date_str1, date_str2):
        # 把字符串日期转成 date 对象（匹配你的索引）
        date1 = datetime.strptime(date_str1, '%Y-%m-%d').date()
        date2 = datetime.strptime(date_str2, '%Y-%m-%d').date()
        # ================= 日前竞价空间 =================
        day1 = feat.loc[date1]
        day2 = feat.loc[date2]
        bid_cols = [f"day_日前竞价空间{i}" for i in range(96)]
        bid1 = day1.loc[bid_cols].values
        bid2 = day2.loc[bid_cols].values
        x_bid = [i for i in range(96)]
        # ================= 价差 =================
        spread1 = spread.loc[date1].values
        spread2 = spread.loc[date2].values
        x_spread = [t.strftime("%H:%M") for t in spread.columns]

        # ================= 辐照 =================
        irr_cols = [f"day_辐照{i}" for i in range(24)]
        irr1 = day1.loc[irr_cols].values
        irr2 = day2.loc[irr_cols].values
        x_irr = [i for i in range(24)]

        # ================= 风速 =================
        wind_cols = [f"day_风速{i}" for i in range(24)]
        wind1 = day1.loc[wind_cols].values
        wind2 = day2.loc[wind_cols].values
        x_wind = [i for i in range(24)]

        # =================  画subplot =================
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        # 大图：跨3列，作为“果”
        ax1 = fig.add_subplot(gs[0, :])
        # 小图：下排3个“因”
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])


        # 子图1：价差
        ax1.plot(x_spread, spread1, label=date_str1, marker='o', linewidth=1.5)
        ax1.plot(x_spread, spread2, label=date_str2, marker='s', linewidth=1.5)
        ax1.set_title("价差对比", fontsize=14)
        ax1.set_ylabel("竞价空间")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xticks(ticks=range(0,48,2))
        ax1.set_xticklabels(labels=x_spread[::2], rotation=45, ha='right', fontsize=10)

        # 子图2：竞价空间
        ax2.plot(x_bid, bid1, label=date_str1, marker='o', linewidth=1.5)
        ax2.plot(x_bid, bid2, label=date_str2, marker='s', linewidth=1.5)
        ax2.set_title("日前竞价空间对比 (0~95)", fontsize=14)
        ax2.set_ylabel("竞价空间")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_xticks(ticks=range(0, 96, 8))
        ax2.set_xticklabels(labels=np.array(x_bid[::8]) // 4, rotation=90, ha='right', fontsize=10)

        # 子图3：辐照
        ax3.plot(x_irr, irr1, label=date_str1, marker='o', linewidth=1.5)
        ax3.plot(x_irr, irr2, label=date_str2, marker='s', linewidth=1.5)
        ax3.set_title("辐照对比 (0~23)", fontsize=14)
        ax3.set_ylabel("辐照")
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 子图4：风速
        ax4.plot(x_wind, wind1, label=date_str1, marker='o', linewidth=1.5)
        ax4.plot(x_wind, wind2, label=date_str2, marker='s', linewidth=1.5)
        ax4.set_title("风速对比 (0~23)", fontsize=14)
        ax4.set_xlabel("时段")
        ax4.set_ylabel("风速")
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout(pad=3.0)
        plt.show()
