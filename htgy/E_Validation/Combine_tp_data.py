import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

def generate_SOC_Mineralization_Data(
    csv_60=PROCESSED_DIR / "ERA5_extracted_range_0594_0606.csv",
    csv_90=PROCESSED_DIR / "ERA5_extracted_range_0892_0908.csv",
    csv_120=PROCESSED_DIR / "ERA5_extracted_range_1188_1212.csv",
    output_csv=DATA_DIR / "SOC_Mineralization_Data.csv"
):
    """
    从 Extract_tp.py 输出的三个 CSV 文件中，构造一个能被 Validation_Mineralization.py
    正常读取和使用的 SOC_Mineralization_Data.csv。
    
    注意：此示例脚本仅演示基本数据处理思路；实际数据列名、行数、
         时间/实验天数等需根据真实情况调整！
    """
    # 1. 分别读入三个CSV，对应三类降雨强度
    df_60  = pd.read_csv(csv_60)
    df_90  = pd.read_csv(csv_90)
    df_120 = pd.read_csv(csv_120)
    
    # ------------------------------------------------------------------------
    # 2. 为了模仿 Validation_Mineralization.py 的做法，我们假设：
    #    - 这三类数据各自只需要 5 行（对应 .iloc[1:6]）
    #    - 其中 Region=Erosion Area 或 Sedimentation Area 是分类标签
    #    - 我们想把它们展开成两列: "Erosion Area", "Sedimentation Area"
    #    - 并把 60 mm/h 的5行放在前，90 mm/h 的5行放中间，120 mm/h 的5行放后面
    #      这样 Validation_Mineralization.py 才可以用 .iloc[1:6] / .iloc[6:11] / .iloc[11:16]
    #
    # 注意：如果你的 CSV 行数不足或不止5行，需在这里进行更灵活的处理/聚合/汇总。
    # ------------------------------------------------------------------------
    
    # 定义一个帮助函数，把 Region=Erosion Area / Sedimentation Area 这2种行展开成2列
    # 并只保留想要的行数(这里示例只取前5行做演示)
    def pivot_erosion_sedimentation(df_in, n_rows=5):
        # 假设 df_in 中至少包含这几列
        # ["LAT","LON","Region","TOTAL_PRECIP","year","month", ...]
        # 以及 "Region" 只会是 "Erosion Area" 或 "Sedimentation Area"
        
        # 先选出我们真正关心的列（如果还需要别的列，也可以加）
        # 这里最关键的是 Region 和我们要分析的某个数值（比如TOTAL_PRECIP，或别的“面积”）
        # 但 Validation_Mineralization.py 其实是直接读取 [“Erosion Area”] / [“Sedimentation Area”]
        # 所以我们需要手动给每个 region 对应一些数值。假设这里就用 df_in["TOTAL_PRECIP"] 演示一下。
        
        # 先截取前 n_rows 行，保证跟 .iloc[1:6] 对应
        df_in = df_in.iloc[:n_rows].copy()
        
        # 做一个简单的 pivot（Region 变成列名，value 用 TOTAL_PRECIP）
        df_pivot = df_in.pivot(columns="Region", values="TOTAL_PRECIP")
        
        # 如果原 df_in 里，Region 同时含 Erosion Area / Sedimentation Area，
        # 那么 pivot 之后就会出现 2 列：“Erosion Area”, “Sedimentation Area”
        # 但因为 pivot 后索引来自原 df_in 的行号，需要 reset_index 变成普通 DataFrame
        df_pivot = df_pivot.reset_index(drop=True)
        
        # 若 pivot 里只出现了 Erosion Area 这一列，没有 Sedimentation Area，
        # 需要先补一列空值，防止后续合并时报错
        if "Erosion Area" not in df_pivot.columns:
            df_pivot["Erosion Area"] = np.nan
        if "Sedimentation Area" not in df_pivot.columns:
            df_pivot["Sedimentation Area"] = np.nan
        
        return df_pivot[["Erosion Area","Sedimentation Area"]]
    
    df_pivot_60  = pivot_erosion_sedimentation(df_60,  n_rows=5)
    df_pivot_90  = pivot_erosion_sedimentation(df_90,  n_rows=5)
    df_pivot_120 = pivot_erosion_sedimentation(df_120, n_rows=5)
    
    # 3. 把它们按行顺序拼起来。这样就形成了 15 行的数据：
    #    前 5 行 → 60 mm/h
    #    中 5 行 → 90 mm/h
    #    后 5 行 → 120 mm/h
    combined_df = pd.concat([df_pivot_60, df_pivot_90, df_pivot_120],
                            ignore_index=True)
    
    # 4. Validation_Mineralization.py 在读取时，会用 .iloc[1:6], .iloc[6:11], .iloc[11:16]
    #    如果 combined_df 总共只有15行 (index=0~14)，那在 .iloc[11:16] 里仍能读到 index=11,12,13,14
    #    这样就足够啦。但是注意 .iloc[1:6] 会跳过 index=0 ！
    #
    #    如果你想跟脚本的切片 1:6 / 6:11 / 11:16 精确对应，就需要再插一行“占位”的假数据放在开头( index=0 )。
    #    举例：我们加一个全空行，这样 row=0 是空的，row=1~5 就是 60 mm/h 的 5 行。
    placeholder_row = pd.DataFrame({"Erosion Area":[np.nan],
                                    "Sedimentation Area":[np.nan]})
    
    final_df = pd.concat([placeholder_row, combined_df], ignore_index=True)
    
    # 这时 final_df 的行数=16 (index 0~15):
    #   index=1..5 -> 60 mm/h
    #   index=6..10 -> 90 mm/h
    #   index=11..15 -> 120 mm/h
    # 与 Validation_Mineralization.py 的 slicing 相吻合.
    
    # 5. 导出到 CSV 文件
    final_df.to_csv(output_csv, index=False)
    print(f"成功生成 {output_csv} (共 {len(final_df)} 行).")


if __name__ == "__main__":
    generate_SOC_Mineralization_Data()
