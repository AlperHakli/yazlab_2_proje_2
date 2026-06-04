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
exclude_columns = [
    "datetime",
    "anomaly",
    "changepoint",
    "source_file",
    "source_group"
]

normalized_df = DataPreprocessing.df_normalization(
    skab_df,
    exclude_columns
)

pca_df = DataPreprocessing.principal_component_analysis(
    normalized_df,
    n_components=1,
    except_columns=exclude_columns
)

from automata.pipeline import AutomataPipeline

pc1_series = pca_df["PC1"].values

train_size = int(len(pc1_series) * 0.8)

train_series = pc1_series[:train_size]
test_series = pc1_series[train_size:]

automata_pipeline = AutomataPipeline(
    paa_window_size=4,
    pattern_window_size=4,
    alphabet_size=3
)

fit_result = automata_pipeline.fit(train_series)
predictions = automata_pipeline.predict(test_series)
# automata test  
#print("State sayısı:", len(fit_result["states"]))
#print("Transition sayısı:", len(fit_result["transition_table"]))
#print("İlk 5 açıklama:")
#for prediction in predictions[:5]:
#    print(prediction)




from evaluation.metrics import EvaluationMetrics

y_test = pca_df["anomaly"].values[train_size:]

results = EvaluationMetrics.evaluate(y_test, predictions)

print("Evaluation Results:")
print(results)




