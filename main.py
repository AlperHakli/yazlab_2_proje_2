import pandas as pd
import json
from settings import settings
from fetch_data import FetchData
from experiments.otomata_runner import run_skab_experiments, run_batadal_experiments
from automata.pipeline import AutomataPipeline
from experiments.cross_dataset_runner import run_cross_dataset_experiments
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from automata.pipeline import AutomataPipeline
from experiments.models import LSTMModel, GRUModel
from experiments.preprocessing import DeepLearningPreprocessor

# Derin öğrenme deney motorlarının projeye dahil edilmesi
from experiments.deep_learning_runner import run_dl_skab_experiments, run_dl_batadal_experiments


def main():
    print("="*20)
    print("yazlab-2 kara kutu ve otomata modeli başlatıldı...")
    print("="*20)

    # --- 1. VERİ OKUMA KATMANI ---
    print("\n veri setleri diskten yükleniyor...")
    skab_v1 = FetchData.fetch_data_from_folder(settings.DATA_PATH, settings.SKAB_VALVE1_PATH, sep=";")
    skab_v2 = FetchData.fetch_data_from_folder(settings.DATA_PATH, settings.SKAB_VALVE2_PATH, sep=";")
    skab_df = pd.concat([skab_v1, skab_v2], ignore_index=True)

    batadal_df = pd.read_csv(settings.BATADAL_PATH, skipinitialspace=True)

    # sembolik otomata deneyleri grid search ile
    #pca
    #paa
    #sax, sliding window, automata, transition probability, noise test, parametre analizi
    print("\nsembolik otomata deneyleri Başlatılıyor...")
    print("-" * 50)
    run_skab_experiments(skab_df)
    run_batadal_experiments(batadal_df)

    # lstm ve gru modelleri uygulanır
    print("\nderin öğrenme yani kara kutu deneyleri başlatılıyor...")
    print("-" * 50)

    # SKAB ve BATADAL için LSTM Model deneyleri
    print("\nLSTM modelleri eğitiliyor")
    run_dl_skab_experiments(skab_df, model_type="LSTM")
    run_dl_batadal_experiments(batadal_df, model_type="LSTM")

    # SKAB ve BATADAL için GRU model deneyleri
    print("\nGRU modelleri eğitiliyor...")
    run_dl_skab_experiments(skab_df, model_type="GRU")
    run_dl_batadal_experiments(batadal_df, model_type="GRU")

    # çıktı üretme kısmı
    print("\n" + "=" * 25 + "açıklanabilirlik kısmı" + "=" * 25)
    print("örnek bir açıklama üretiliyor ... ")

    exclude_cols = ["datetime", "anomaly", "changepoint", "source_file", "source_group"]
    sensor_cols = [col for col in skab_df.columns if col not in exclude_cols]

    train_size = int(len(skab_df) * 0.6) # %60 training

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
# 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(skab_df[sensor_cols].iloc[:train_size].values)
    X_test_scaled = scaler.transform(skab_df[sensor_cols].iloc[train_size:].values)
# çok boyutlu sensör verisi tek boyuta indiriliyor.
    pca = PCA(n_components=1)
    X_train_pc1 = pca.fit_transform(X_train_scaled).flatten()
    X_test_pc1 = pca.transform(X_test_scaled).flatten()
   # paa → sax→ pattern extraction → automata
    demo_pipeline = AutomataPipeline(paa_window_size=4, pattern_window_size=4, alphabet_size=3)
    demo_pipeline.fit(X_train_pc1)

    demo_pipeline.explainer.reset_path()
    explanations = demo_pipeline.predict(X_test_pc1)

    # json formatı ile çıktı
    print("\n ilk anomali çıktısı diğer çıktılar detaylı loglarda mevcut")
    print("-" * 60)
    # ilk anomali örneğini json formatında bas
    anomaly_sample = next((item for item in explanations if item["decision"] == "anomaly"), explanations[0])
    print(json.dumps(anomaly_sample, indent=4))
    print("-" * 60)

    # tablo formatı şeklinde çıktı
    print("\nilk 10 zaman adımının özet tablosu")
    df_table = pd.DataFrame(explanations[:10])
    columns_to_show = ["state", "pattern", "status", "mapped_to", "probability", "path_probability", "decision",
                       "distance"]
    print(df_table[columns_to_show].to_string(index=True))
    print("=" * 78)

    print("\n=====================================================")
    print("tüm süreç başarıyla bitti otomata ve derin öğrenme logları hazır!")
    print("="*20)
    print("\nCross-dataset deneyleri başlatılıyor...")
    run_cross_dataset_experiments(skab_df, batadal_df)

# Bu main dosyası veri setlerini yükler, automata ve derin öğrenme deneylerini çalıştırır,
#  açıklanabilirlik çıktısı üretir ve veri setleri arası genellenebilirlik testlerini başlatır.


if __name__ == "__main__":
    main()

