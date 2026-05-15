import os
from pathlib import Path
class Settings():
    """Ana ayarları ve dosya yollarını içeren sınıf"""


    # ana dizine götürür
    BASE_PATH = Path(__file__).resolve().parent
    # verilerin bulunduğu klasörün yolu
    DATA_PATH = BASE_PATH / "data"
    # skab valve 1 yolu
    SKAB_VALVE1_PATH =  "skab_valve1"
    #skab valve 2 yolu
    SKAB_VALVE2_PATH = "skab_valve2"
    # batadal dataset yolu
    BATADAL_PATH = DATA_PATH / "batadal_dataset.csv"


settings = Settings()



