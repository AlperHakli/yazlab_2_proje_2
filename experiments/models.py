import tensorflow as tf
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)

class LSTMModel(tf.keras.Model):
    """Derin Öğrenme Zaman Serisi Anomali Sınıflandırma LSTM Modeli"""
    def __init__(self, units=64, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        logger.info(f"LSTM Modeli {units} birim ile inşa ediliyor.")
        self.lstm = layers.LSTM(units=units, return_sequences=False)
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.fullyconnected = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dropout(x, training=training)
        return self.fullyconnected(x)

class GRUModel(tf.keras.Model):
    """Derin Öğrenme Zaman Serisi Anomali Sınıflandırma GRU Modeli"""
    def __init__(self, units=64, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        logger.info(f"GRU Modeli {units} birim ile inşa ediliyor.")
        self.gru = layers.GRU(units=units, return_sequences=False)
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.fullyconnected = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.gru(inputs)
        x = self.dropout(x, training=training)
        return self.fullyconnected(x)