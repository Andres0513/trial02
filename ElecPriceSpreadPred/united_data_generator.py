import os
import glob
import pandas as pd
from enum import Enum
import holidays


# 电网工况枚举
class grid_env(Enum):
    NORMAL = 1
    GAS_IN_MARKET = 2   # 气电入市


def postprocess_merged_weather_data(merged_data: pd.DataFrame) -> pd.DataFrame:
    df = merged_data.copy()
    # 1. 除了第一列（时间），其他列整体上移一行（相当于把上一行的值“推”到下一行）
    cols_to_shift = df.columns[1:]  # 除了第一列
    df[cols_to_shift] = df[cols_to_shift].shift(-1)
    # 2. 去掉最后一行
    df = df.drop(index=df.index[-1]).reset_index(drop=True)
    def fix_time(time_str):
        time_str = str(time_str).strip()
        if "24:00" in time_str:
            date_part = time_str.split(" ")[0]
            new_date = pd.to_datetime(date_part) + pd.Timedelta(days=1)
            return new_date.strftime("%Y-%m-%d 00:00")
        return time_str
    df['时间'] = df['时间'].apply(fix_time)
    return df

# 读取和处理天气
def load_and_postprocess_weather_data(search_path, sub_folder):
    abs_path = os.path.join(search_path, sub_folder)
    # ========== 这部分文件是<=20260519的天气数据，是历史真实天气数据。除了读取，这部分数据还要做些后处理（历史原因） ==========
    # 查找所有 .xlsx 且 包含“天气” 的文件
    excel_files = [
        f for f in glob.glob(os.path.join(abs_path, "*.xlsx"))
        if "天气" in os.path.basename(f)
    ]

    # 记录df
    df_list = []
    for file in excel_files:
        print("正在读取：", os.path.basename(file))
        df = pd.read_excel(file)
        df_list.append(df)
    # 合并所有的DataFrame
    df_merged = pd.concat(df_list, ignore_index=True)
    df_merged = df_merged.sort_values(by='时间', ascending=True)
    df_1 = postprocess_merged_weather_data(df_merged)

    # ========== 这部分文件是>20260519的天气数据，尽量用历史预测天气数据 ==========
    # 其中，0607用的是可能是真实值。那天忘记拿预测数据了。不是7号忘了，也是6号或5号
    # 0601之前，预报值中的24点，用的是23点的数据填充的，当时没意识到24点的数据不在当天日期中。
    # 记录df
    df_list = []
    excel_files = [
        f for f in glob.glob(os.path.join(abs_path, "*预报值*.xlsx"))
    ]
    for file in excel_files:
        print("正在读取: ", os.path.basename(file))
        df = pd.read_excel(file)
        # 为了兼容，重命名
        df = df.rename(columns={
            "浙江省Ecmwf温度": "温度",
            "浙江省Ecmwf相对湿度": "湿度",
            "浙江省Ecmwf降雨量": "降雨",
            "浙江省Ecmwf风速": "风速",
            "浙江省Ecmwf辐照": "辐照",
            "浙江省Ecmwf云量": "云"
        })
        # 为了兼容，重命名
        df = df.rename(columns={
            "温度（浙江省ECMWF预测值）": "温度",
            "相对湿度（浙江省ECMWF预测值）": "湿度",
            "降雨量（浙江省ECMWF预测值）": "降雨",
            "风速（浙江省ECMWF预测值）": "风速",
            "辐照（浙江省ECMWF预测值）": "辐照",
            "云量（浙江省ECMWF预测值）": "云"
        })
        df_list.append(df)

    # 合并所有的DataFrame
    df_2 = pd.concat(df_list, ignore_index=True)
    df_2['时间'] = df_2['时间'].apply(fix_time)
    df_final = pd.concat([df_1, df_2], ignore_index=True)
    df_final = df_final.sort_values(by='时间', ascending=True).reset_index(drop=True)
    df_final['时间'] = pd.to_datetime(df_final['时间'])
    return df_final

# ==============================
# ✅ 正确修复 24:00 时间格式
# 例如把 2024-10-01 24:00 → 2024-10-02 00:00
# ==============================
def fix_time(time_str):
    time_str = str(time_str).strip()
    if "24:00" in time_str:
        date_part = time_str.split(" ")[0]
        new_date = pd.to_datetime(date_part) + pd.Timedelta(days=1)
        return new_date.strftime("%Y-%m-%d 00:00")
    return time_str

# ===================== 读取电价 =====================
def load_electricity_clearing_data(search_path, sub_folder):
    folder_path = os.path.join(search_path, sub_folder)

    df_list = []

    # 遍历文件夹里所有 xlsx 文件
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx") and "现货出清数据" in file:
            print(f"正在读取：{file}")

            file_full_path = os.path.join(folder_path, file)

            # ✅ 关键：跳过前6行，从第7行开始读真正数据
            df = pd.read_excel(file_full_path, skiprows=range(1,6),header=0)

            # 提取所需列
            df = df[["时间", "日前价格", "实时价格", "价差（实时-日前）"]].copy()

            df_list.append(df)

    # 合并所有文件
    df_total = pd.concat(df_list, ignore_index=True)

    df_total["时间"] = df_total["时间"].apply(fix_time)
    df_total["时间"] = pd.to_datetime(df_total["时间"])

    # 排序
    df_total = df_total.sort_values("时间").reset_index(drop=True)

    print("\n✅ 出清数据合并完成！")
    print(df.head())
    print(f"\n出清总数据条数：{len(df)}")

    return df_total

# ===================== 读取竞价空间 =====================
def load_electricity_bidding_space_data(search_path, sub_folder):
    folder_path = os.path.join(search_path, sub_folder)

    df_list = []

    # 遍历文件夹里所有 xlsx 文件
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx") and "竞价空间数据" in file:
            print(f"正在读取：{file}")

            file_full_path = os.path.join(folder_path, file)

            # ✅ 关键：跳过前6行，从第7行开始读真正数据
            df = pd.read_excel(file_full_path, skiprows=range(1,5),header=0)

            # 提取所需列
            df = df[["时间",	"日前竞价空间", "实际竞价空间","统调负荷预测","统调负荷实际","外来电计划","外来电实际","光伏出力预测",
                     "光伏出力实际","风电出力预测","风电出力实际","固定出力计划"]].copy()

            df_list.append(df)

    # 合并所有文件
    df_total = pd.concat(df_list, ignore_index=True)

    df_total["时间"] = df_total["时间"].apply(fix_time)
    df_total["时间"] = pd.to_datetime(df_total["时间"])

    # 排序
    df_total = df_total.sort_values("时间").reset_index(drop=True)

    print("\n✅ 竞价数据合并完成！")
    print(df.head())
    print(f"\n竞价总数据条数：{len(df)}")

    return df_total


#-----------生成电网环境的df----------
def genrate_env_flag(date_str):
    d = pd.to_datetime(date_str)
    d_date = d.normalize()
    # 2024-05-01 ~ 2024-12-31 → GAS
    # 2026-04-01 ~ 2026-05-02 → GAS
    # 其余 → NORMAL
    if (pd.Timestamp("2024-05-01") <= d_date <= pd.Timestamp("2024-12-31")) or \
            (pd.Timestamp("2026-04-01") <= d_date):
        return grid_env.GAS_IN_MARKET.value
    else:
        return grid_env.NORMAL.value

def generate_forward_feature(df_1, df_2, df_3):
    # =======================处理竞价空间数据============================
    # 初始化新列，构建竞价空间特征数据
    for i in range(8):
        df_3[f'日前竞价空间{i}'] = pd.NA
        df_3[f'统调负荷预测{i}'] = pd.NA
        df_3[f'外来电计划{i}'] = pd.NA
        df_3[f'光伏出力预测{i}'] = pd.NA
        df_3[f'风电出力预测{i}'] = pd.NA
        df_3[f'固定出力计划{i}'] = pd.NA
    # 遍历出清时间点，填充竞价空间数据
    for idx, target_time in df_3['时间_dt'].items():
        # 筛选时间点前两小时内的竞价数据
        mask = (df_1['时间_dt'] > target_time - pd.Timedelta(hours=2)) & (df_1['时间_dt'] <= target_time)
        window_data = df_1.loc[mask, ['时间_dt', '日前竞价空间', '统调负荷预测', '外来电计划', '光伏出力预测', '风电出力预测', '固定出力计划']].copy()
        # 按时间从近到远倒序排序，这样第一个元素就是距离target_time最近的
        window_data = window_data.sort_values('时间_dt', ascending=False).reset_index(drop=True)
        # 取往前两小时内的数据，不足的保留NA
        for i in range(min(8, len(window_data))):
            df_3.at[idx, f'日前竞价空间{i}'] = window_data['日前竞价空间'].iloc[i]
            df_3.at[idx, f'统调负荷预测{i}'] = window_data['统调负荷预测'].iloc[i]
            df_3.at[idx, f'外来电计划{i}'] = window_data['外来电计划'].iloc[i]
            df_3.at[idx, f'光伏出力预测{i}'] = window_data['光伏出力预测'].iloc[i]
            df_3.at[idx, f'风电出力预测{i}'] = window_data['风电出力预测'].iloc[i]
            df_3.at[idx, f'固定出力计划{i}'] = window_data['固定出力计划'].iloc[i]
    # =======================处理竞价天气数据============================
    # 初始化新列，构建天气特征数据
    for i in range(2):
        df_3[f'温度{i}'] = pd.NA
        df_3[f'湿度{i}'] = pd.NA
        df_3[f'降雨{i}'] = pd.NA
        df_3[f'辐照{i}'] = pd.NA
        df_3[f'云{i}'] = pd.NA
        df_3[f'风速{i}'] = pd.NA

    # 遍历出清时间点，填充天气数据
    for idx, target_time in df_3['时间_dt'].items():
        # 筛选时间点前两小时内的天气数据
        mask = (df_2['时间_dt'] > target_time - pd.Timedelta(hours=2)) & (df_2['时间_dt'] <= target_time)
        window_data = df_2.loc[mask, ['时间_dt', '温度', '湿度', '降雨', '辐照', '云', '风速']]

        # 按时间从近到远倒序排序，这样第一个元素就是距离target_time最近的
        window_data = window_data.sort_values('时间_dt', ascending=False).reset_index(drop=True)

        # 往前取两小时内的数据，不足的保留NA
        for i in range(min(2, len(window_data))):
            df_3.at[idx, f'温度{i}'] = window_data['温度'].iloc[i]
            df_3.at[idx, f'湿度{i}'] = window_data['湿度'].iloc[i]
            df_3.at[idx, f'降雨{i}'] = window_data['降雨'].iloc[i]
            df_3.at[idx, f'辐照{i}'] = window_data['辐照'].iloc[i]
            df_3.at[idx, f'云{i}'] = window_data['云'].iloc[i]
            df_3.at[idx, f'风速{i}'] = window_data['风速'].iloc[i]
    return df_3

def generate_that_day_feature(df_1, df_2, df_3):
    # =======================处理竞价空间数据============================
    # 初始化新列，构建竞价空间特征数据
    for i in range(4*24):
        df_3[f'day_日前竞价空间{i}'] = pd.NA
        df_3[f'day_统调负荷预测{i}'] = pd.NA
        df_3[f'day_外来电计划{i}'] = pd.NA
        df_3[f'day_光伏出力预测{i}'] = pd.NA
        df_3[f'day_风电出力预测{i}'] = pd.NA
        df_3[f'day_固定出力计划{i}'] = pd.NA
    # 遍历出清时间点，填充竞价空间数据
    for idx, target_time in df_3['时间_dt'].items():
        # 筛选当天的竞价数据
        mask = (df_1['时间_dt'].dt.date==target_time.date())
        window_data = df_1.loc[mask, ['时间_dt', '日前竞价空间', '统调负荷预测', '外来电计划', '光伏出力预测', '风电出力预测', '固定出力计划']].copy()
        # 当天数据不全的，直接跳过
        if len(window_data) != 4*24:
            continue
        for i in range(4*24):
            df_3.at[idx, f'day_日前竞价空间{i}'] = window_data['日前竞价空间'].iloc[i]
            df_3.at[idx, f'day_统调负荷预测{i}'] = window_data['统调负荷预测'].iloc[i]
            df_3.at[idx, f'day_外来电计划{i}'] = window_data['外来电计划'].iloc[i]
            df_3.at[idx, f'day_光伏出力预测{i}'] = window_data['光伏出力预测'].iloc[i]
            df_3.at[idx, f'day_风电出力预测{i}'] = window_data['风电出力预测'].iloc[i]
            df_3.at[idx, f'day_固定出力计划{i}'] = window_data['固定出力计划'].iloc[i]
    # =======================处理竞价天气数据============================
    # 初始化新列，构建天气特征数据
    for i in range(24):
        df_3[f'day_温度{i}'] = pd.NA
        df_3[f'day_湿度{i}'] = pd.NA
        df_3[f'day_降雨{i}'] = pd.NA
        df_3[f'day_辐照{i}'] = pd.NA
        df_3[f'day_云{i}'] = pd.NA
        df_3[f'day_风速{i}'] = pd.NA

    # 遍历出清时间点，填充天气数据
    for idx, target_time in df_3['时间_dt'].items():
        # 筛选当天的天气数据
        mask = (df_2['时间_dt'].dt.date==target_time.date())
        window_data = df_2.loc[mask, ['时间_dt', '温度', '湿度', '降雨', '辐照', '云', '风速']]
        if len(window_data) != 24:
            print(f"⚠️【数据缺失告警】行索引:{idx}, 目标时间:{target_time}, "
                  f" 实际条数:{len(window_data)}")
            continue
        # 取当天的天气数据
        for i in range(24):
            df_3.at[idx, f'day_温度{i}'] = window_data['温度'].iloc[i]
            df_3.at[idx, f'day_湿度{i}'] = window_data['湿度'].iloc[i]
            df_3.at[idx, f'day_降雨{i}'] = window_data['降雨'].iloc[i]
            df_3.at[idx, f'day_辐照{i}'] = window_data['辐照'].iloc[i]
            df_3.at[idx, f'day_云{i}'] = window_data['云'].iloc[i]
            df_3.at[idx, f'day_风速{i}'] = window_data['风速'].iloc[i]
    return df_3

def generate_recent_prices_spread(df_3):
    # 先按时间排序
    df_3 = df_3.sort_values("时间").reset_index(drop=True)
    # 提取时间中的「时刻」部分（只保留时:分），用来匹配前几天的同一时间
    df_3["时刻"] = df_3["时间"].dt.strftime("%H:%M")
    #  循环生成前 1~7 天同一时间的价差
    for days in range(2, 8):
        col_name = f"价差_前{days}天"
        # 复制一份数据，把时间往前推 days 天，然后按「时间」+「时刻」合并，就能拿到同一时刻的历史数据
        tmp = df_3[["时间", "时刻", "价差（实时-日前）"]].copy()
        tmp["时间_偏移"] = tmp["时间"] + pd.Timedelta(days=days)
        # 合并：用「偏移后的时间」和「原时间」做 key，匹配同一时刻的历史价差
        merged = pd.merge(
            df_3,
            tmp[["时间_偏移", "时刻", "价差（实时-日前）"]].rename(columns={"价差（实时-日前）": col_name}),
            left_on=["时间", "时刻"],
            right_on=["时间_偏移", "时刻"],
            how="left"
        )
        # 只保留新增的那一列
        df_3[col_name] = merged[col_name]
    # 删掉辅助列
    df_3 = df_3.drop(columns=["时刻"])
    return df_3


def generate_feature_df(bidding_space_df: pd.DataFrame, weather_df: pd.DataFrame, clearing_df: pd.DataFrame, start_date, end_date)->pd.DataFrame:
    # 转时间格式
    df_1 = bidding_space_df.copy()
    df_1['时间_dt'] = pd.to_datetime(df_1['时间'])
    df_2 = weather_df.copy()
    df_2['时间_dt'] = pd.to_datetime(df_2['时间'])
    df_3 = clearing_df.copy()
    df_3['时间_dt'] = pd.to_datetime(df_3['时间'])

    df_3 = generate_recent_prices_spread(df_3)
    df_3 = generate_forward_feature(df_1, df_2, df_3)
    df_3 = generate_that_day_feature(df_1, df_2, df_3)


    df_3['日期']= df_3['时间'].dt.date
    df_3 = df_3[(df_3['日期'] >= pd.to_datetime(start_date).date()) & (df_3['日期'] <= pd.to_datetime(end_date).date())]

    # 电网工况
    df_3['grid_env'] = df_3['时间'].apply(genrate_env_flag)

    # 星期几（0=周一，6=周日）
    df_3['星期'] = df_3['时间_dt'].dt.weekday  # 0-6

    # 第几季度
    df_3['季度'] = df_3['时间_dt'].dt.quarter  # 1-4

    # 时间
    df_3['clock'] = (((df_3['时间_dt'].dt.hour) + (df_3['时间_dt'].dt.minute) / 60.0) * 2.0).astype(int)

    # 是否节假日（中国节假日，含周末）
    cn_holidays = holidays.China()
    df_3['是否节假日'] = df_3['时间_dt'].dt.date.apply(lambda x: x in cn_holidays or x.weekday() >= 5).astype(int)
    df_3['是否工作日'] = 1-df_3['是否节假日']

    df_3 = df_3.drop(columns=['日期', '时间_dt'])

    return df_3


if __name__ == '__main__':
    # 自动获取当前脚本所在目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 往上一级
    BASE_DIR = os.path.dirname(BASE_DIR)
    # 原始文件夹名
    ORIGIN_FOLDER = "united_data_resources"
    # 输出文件夹名
    TARGET_FOLDER = 'electricity_data'

    start_date = '2024-05-02'
    end_date = '2026-06-12'


    # 拼接路径：项目目录/目标文件夹
    search_path = os.path.join(BASE_DIR, ORIGIN_FOLDER)
    target_path = os.path.join(os.path.join(BASE_DIR, TARGET_FOLDER), 'united' + '.xlsx')

    # =========== 读取三大数据：天气、竞价空间、出清价格 ============
    weather_df = load_and_postprocess_weather_data(search_path, '天气数据')
    clearing_df = load_electricity_clearing_data(search_path, "现货出清数据")
    bidding_space_df = load_electricity_bidding_space_data(search_path, "竞价空间数据")
    # 按日期划分一下工况
    bidding_space_df['grid_env'] = bidding_space_df['时间'].apply(genrate_env_flag)
    uninted_df = generate_feature_df(bidding_space_df, weather_df, clearing_df, start_date, end_date)
    uninted_df.to_csv("uninted_df.csv")
    print(f"✅ 数据集已保存：uninted_df.csv")

    # KEY_WORDS = ["天气"]
    # for key_word in KEY_WORDS:
    #     # 拼接路径：项目目录/目标文件夹
    #     search_path = os.path.join(BASE_DIR, ORIGIN_FOLDER)
    #     target_path = os.path.join(os.path.join(BASE_DIR, TARGET_FOLDER), key_word+'.xlsx')
    #     df_final = load_all_files_in_folder_by_keyword(search_path, key_word, weather_prediction=True)
    #     # 原文件数据有问题，后处理下
    #     if key_word == "出清":    # 处理出清数据，读的是天天智电后来给的，最后模型没用这份数据，模型用的是自己下载的那份
    #         df_final = postprocess_merged_clearing_data(df_final)
    #     if key_word == "竞价空间":  # 处理竞价空间数据，读的是天天智电后来给的，最后模型没用这份数据，模型用的是自己下载的那份
    #         df_final = postprocess_merged_bidding_space_data(df_final)
    #     if key_word == "天气":    # 处理天气数据，读的是天天智电后来给的，模型用的是这份数据。
    #         df_final = postprocess_merged_weather_data(df_final)
    #     df_final.to_excel(target_path, index=False) # 输出excel文件，文件名是key_words加xlsx后缀
