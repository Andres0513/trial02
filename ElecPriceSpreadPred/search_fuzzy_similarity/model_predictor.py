import pandas as pd
from enums import similarity_method
from fuzzy_similarity_data import Fuzzy_Similarity_Data
from datetime import date, timedelta, time
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Model_Predictor:
    def __init__(self):
        return

    def run(self, train_df, val_df):
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

