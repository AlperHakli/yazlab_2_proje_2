import os
import pandas as pd
from pandas import DataFrame
from pathlib import Path


class FetchData():
    @staticmethod
    def fetch_data_from_folder(base_path, file_path: str, sep:str) -> DataFrame:
        """
        Verilen dosya yolundan tek tek .csv dosyalarını çekip hepsini concat ile birleştirir
        :param full_path: İlgili dosya yolu
        :param base_path: dosyanın ana klasör yolu
        :param sep: csv columnlarının hangi karakter ile ayrıldığını belirtir
        """
        full_path = Path(base_path / file_path)
        # .csv ile biten tüm dosyaları alır
        df_list = []
        # tek tek her dosya için dosya alınıp csv ye çevrilir ardından source_file ve source_group eklenip df_list e eklenir
        for file in full_path.glob("*.csv"):
            source_file = file.name
            df = pd.read_csv(file , sep=sep)
            df["source_file"] = source_file
            df["source_group"] = file_path
            df_list.append(df)

        if not df_list is None:
            main_df = pd.concat(df_list, ignore_index=True)
            return main_df
