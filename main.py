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










