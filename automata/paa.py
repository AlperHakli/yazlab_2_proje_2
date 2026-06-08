import numpy as np


class PAA:
    """
    Piecewise Aggregate Approximation.
    Zaman serisini parçalara böler ve her parçanın ortalamasını alır.
    """
    # pencere boyutunu kaydeder
    def __init__(self, window_size: int = 4):
        self.window_size = window_size

    def transform(self, series):
        series = np.array(series, dtype=float)

        paa_values = []

        for i in range(0, len(series), self.window_size):
            window = series[i:i + self.window_size]

            if len(window) == self.window_size:
                paa_values.append(np.mean(window))

        return np.array(paa_values)
    
    #PAA, zaman serisini belirli büyüklükte pencerelere ayırır ve her pencerenin ortalamasını alarak veriyi sıkıştırır.
    #  Böylece hem veri boyutu azalır hem de gürültü etkisi azaltılır.
    #  Biz projemizde PCA sonrası elde edilen PC1 zaman serisini PAA ile özetleyerek SAX aşamasına hazırladık