import sklearn
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)
class DataPreprocessing():

    @staticmethod
    def _split_dataframe(dataframe: pd.DataFrame , except_columns: list[str])->tuple:
        """Verilen dataframe yi except_columns daki listeye göre ikiye ayırıp ayrı ayrı döner \n
        ilk değer: except_columns çıkarılmış dataframe \n
        ikinci değer: except_columns lardan yeni bir dataframe
        """
        except_df = dataframe.get(except_columns)
        main_df = dataframe.drop(columns=except_columns , errors='ignore')

        return (main_df , except_df)

    @staticmethod
    def _df_normalization(dataframe: pd.DataFrame)->pd.DataFrame:
        """
        Veriyi ortalaması 0 standart sapması 1 olcak şekilde dönüştürür
        :param dataframe: ilgili dataframe
        :return: işlenmiş dataframe
        """

        #normalizasyon işlemi
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(dataframe)
        return pd.DataFrame(
            normalized_values,
            columns=dataframe.columns,
            index=dataframe.index  # Indexleri korumak hayati önem taşır
        )

    @staticmethod
    def df_normalization(dataframe: pd.DataFrame , except_columns: list[str])->pd.DataFrame:
        """
        Veriyi except_columns daki columnlar harici ortalaması 0 standart sapması 1 olcak şekilde dönüştürür
        """
        logging.info("Veriye normalizasyon uygulanıyor ... ")
        # parçalama işlemi
        main_db , except_db = DataPreprocessing._split_dataframe(dataframe=dataframe , except_columns=except_columns)
        # veriyi işler
        scaled_df = DataPreprocessing._df_normalization(dataframe=main_db)
        logging.info("Veriye normalizasyon uygulandı")
        # harici veri ve işlenmiş veri birleştirilir
        return pd.concat([scaled_df , except_db] , axis=1)


    @staticmethod
    def _principal_component_analysis(dataframe: pd.DataFrame , n_components: int)->pd.DataFrame:
        """Verilen dataframe de pca işlemi uygulayıp boyut indirger boyunu n_components kadar yapar"""
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(dataframe)
        column_names = [f"PC{i + 1}" for i in range(n_components)]
        return pd.DataFrame(pca_result, columns=column_names, index=dataframe.index)

    @staticmethod
    def principal_component_analysis(dataframe: pd.DataFrame ,n_components: int ,  except_columns: list[str])->pd.DataFrame:
        """
        Verilen dataframe den except_columns listesini çıkarıp kalan verilere\n
         n_components kadar boyut olacak şekilde\n
         boyut indirgeme yapar ve tüm dataframeyi döner
         """
        logging.info("PCA uygulanıyor ...")
        main_db , except_db = DataPreprocessing._split_dataframe(dataframe=dataframe , except_columns=except_columns)

        pca_df = DataPreprocessing._principal_component_analysis(dataframe=main_db , n_components=n_components)

        full_fb = pd.concat([pca_df , except_db], axis=1)

        logging.info("PCA başarıyla uygulandı ...")

        return full_fb




