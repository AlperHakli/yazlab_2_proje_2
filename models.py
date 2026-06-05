from tensorflow.keras import layers
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)
class LSTMModel(tf.keras.Model):
    """Ana lstm modeli"""
    def __init__(self , units , dropout_rate):
        logger.info("lstm modeli başlatıldı")
        super(LSTMModel , self).__init__()
        self.lstm = layers.LSTM(units=units)
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.fullyconnected = layers.Dense(1 , activation="sigmoid")


    def call(self , inputs , training= False):
        logger.info("lstm modeli eğitiliyor ...")
        x = self.lstm(inputs)
        x = self.dropout(x , training=training)
        logger.info("lstm model eğitimi başarılı")
        return self.fullyconnected(x)

class GRUModel(tf.keras.Model):
    def __init__(self , units , dropout_rate):
        logger.info("gru modeli başlatıldı")
        super(GRUModel , self).__init__()
        self.gru = layers.GRU(units=units)
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.fullconnected = layers.Dense(1 , activation="sigmoid")


    def call(self , inputs , training = False):
        logger.info("gru modeli eğitiliyor ...")
        x = self.gru(inputs)
        x = self.dropout(x)
        logger.info("gru model eğitimi başarılı")
        return self.fullconnected(x)
