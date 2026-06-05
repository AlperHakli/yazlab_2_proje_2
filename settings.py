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
    AUTOMATA_THRESHOLD = 0.2

    PARAM_WINDOW_SIZES = [3, 4, 5, 6]
    PARAM_ALPHABET_SIZES = [3, 4, 5, 6]
    RANDOM_SEEDS = [42, 123, 2026, 7, 999]

settings = Settings()



