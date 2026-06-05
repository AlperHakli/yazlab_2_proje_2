import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class Utils:
    def __init__(self):
        ...
    @staticmethod
    def add_gaussian_noise(series: np.ndarray, noise_level: float) -> np.ndarray:
        """zaman serisine 7. ister gereği gaussian gürültüsü ekler."""
        noise = np.random.normal(0, noise_level, series.shape)
        return series + noise
class DeepLearningPreprocessor:
    """
    StandartScaler ve PCA ile veri ön işleme yapar veri sızıntısını önlemek için state tutar
    """

    def __init__(self, n_components=None):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if n_components else None
        self.sensor_cols = None

    def fit(self, df: pd.DataFrame, exclude_columns: list):
        """sadece eğitim verisi üzerinden öğrenir"""
        self.sensor_cols = [col for col in df.columns if col not in exclude_columns]
        X_sensor = df[self.sensor_cols].values

        self.scaler.fit(X_sensor)
        if self.pca:
            X_scaled = self.scaler.transform(X_sensor)
            self.pca.fit(X_scaled)
        logger.info("preprocessor fit işlemi tamamlandı.")

    def transform(self, df: pd.DataFrame, exclude_columns: list) -> tuple:
        """öğrenilen parametreleri sızıntısız bir şekilde veriye uygular"""
        X_sensor = df[self.sensor_cols].values
        y = df["anomaly"].values if "anomaly" in df.columns else df["ATT_FLAG"].values

        X_scaled = self.scaler.transform(X_sensor)
        if self.pca:
            X_features = self.pca.transform(X_scaled)
        else:
            X_features = X_scaled

        return X_features, y

    @staticmethod
    def create_3d_windows(X_data: np.ndarray, y_data: np.ndarray, window_size: int) -> tuple:
        """
        Zaman serisi verisini 3 boyutlu sensör verisine dönüştürür
        """
        X_3d, y_3d = [], []
        for i in range(len(X_data) - window_size):
            X_3d.append(X_data[i: i + window_size])
            # pencerenin son adımındaki etiketi hedef değişken olarak alınır
            y_3d.append(y_data[i + window_size])
        return np.array(X_3d, dtype=np.float32), np.array(y_3d, dtype=np.float32)