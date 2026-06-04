from fetch_data import FetchData
from data_preprocessing import DataPreprocessing
from settings import settings
import pandas as pd

# --- VERİLERİN ÇEKİLDİĞİ KATMAN ---

skab_value_1 = FetchData.fetch_data_from_folder(settings.DATA_PATH,settings.SKAB_VALVE1_PATH , sep=";")
skab_value_2 = FetchData.fetch_data_from_folder(settings.DATA_PATH , settings.SKAB_VALVE2_PATH , sep=";")

# TODO skab dataframe si için anomaly değişkeni hedef değişken iken batadal dataframe si için ATT_FLAG hedef değişken

# --- VERİLERİ TOPLAMA İŞLEMİ ---

skab_df = pd.concat([skab_value_1 , skab_value_2] , ignore_index=True)

# batadal df için veri okunması skipinitialspace özelliği column isimlerindeki boşlukları siler
batadal_df = pd.read_csv(settings.BATADAL_PATH , skipinitialspace = True)
exclude_columns = [
    "datetime",
    "anomaly",
    "changepoint",
    "source_file",
    "source_group"
]

normalized_df = DataPreprocessing.df_normalization(
    skab_df,
    exclude_columns
)

pca_df = DataPreprocessing.principal_component_analysis(
    normalized_df,
    n_components=1,
    except_columns=exclude_columns
)

print(pca_df.head())
print(pca_df.columns.tolist())









