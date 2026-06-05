import pandas as pd
from settings import settings
from fetch_data import FetchData
from experiment_runner import run_skab_experiments, run_batadal_experiments


def main():


    # verilerin çekildiği kısım
    print("\nveri setleri yükleniyor")

    # 1. SKAB Verilerinin Çekilmesi ve Birleştirilmesi
    skab_value_1 = FetchData.fetch_data_from_folder(settings.DATA_PATH, settings.SKAB_VALVE1_PATH, sep=";")
    skab_value_2 = FetchData.fetch_data_from_folder(settings.DATA_PATH, settings.SKAB_VALVE2_PATH, sep=";")
    skab_df = pd.concat([skab_value_1, skab_value_2], ignore_index=True)
    print(f"-> SKAB veri seti hazırlandı boyut {skab_df.shape}")


    batadal_df = pd.read_csv(settings.BATADAL_PATH, skipinitialspace=True)
    print(f"-> batadal veri seti hazırlandı boyut: {batadal_df.shape}")


    print("\ndeneyler başlatıldı")

    # skab ve batadal deneyleri
    run_skab_experiments(skab_df)


    run_batadal_experiments(batadal_df)


    print("tüm işlemler başarıyla bitti sonuc dosyaları hazır.")



if __name__ == "__main__":
    main()