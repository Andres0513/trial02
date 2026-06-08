from enum import Enum

class similarity_method(Enum):
    COSINE = 'COSINE'
    MAE = 'MAE'
    RMSE = 'RMSE'
    PEARSON = 'pearson'
    MAPPED_RMSE = 'MAPPED_RMSE'