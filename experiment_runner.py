import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from settings import settings
from automata.pipeline import AutomataPipeline

from evaluation.metrics import EvaluationMetrics


def run_skab_experiments(skab_df: pd.DataFrame):
    """
    SKAB veri seti için GroupKFold 5-Fold bazlı
    veri sızıntısız parametre analiz döngüsü.
    """
    exclude_cols = ["datetime", "anomaly", "changepoint", "source_file", "source_group"]
    sensor_cols = [col for col in skab_df.columns if col not in exclude_cols]

    groups = skab_df["source_file"].values
    X = skab_df[sensor_cols].values
    y = skab_df["anomaly"].values

    gkf = GroupKFold(n_splits=5)
    experiment_logs = []

    print("\nskab veriseti için analiz başladı")

    for w_size in settings.PARAM_WINDOW_SIZES:
        for a_size in settings.PARAM_ALPHABET_SIZES:
            fold_metrics, fold_state_counts, fold_densities = [], [], []

            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                pca = PCA(n_components=1)
                X_train_pc1 = pca.fit_transform(X_train_scaled).flatten()
                X_test_pc1 = pca.transform(X_test_scaled).flatten()

                pipeline = AutomataPipeline(paa_window_size=w_size, pattern_window_size=w_size, alphabet_size=a_size)
                fit_res = pipeline.fit(X_train_pc1)
                predictions = pipeline.predict(X_test_pc1)

                # --- METRİK BAĞLANTI NOKTASI ---
                metrics_res = EvaluationMetrics.evaluate(y_test, predictions)
                f1 = metrics_res["f1_score"]

                states_count = len(fit_res["states"])
                density = len(fit_res["transition_table"]) / (states_count ** 2 if states_count > 0 else 1)

                fold_metrics.append(f1)
                fold_state_counts.append(states_count)
                fold_densities.append(density)

            log_entry = {
                "window_size": w_size, "alphabet_size": a_size,
                "f1_mean": np.mean(fold_metrics), "f1_std": np.std(fold_metrics),
                "state_count_mean": np.mean(fold_state_counts), "transition_density_mean": np.mean(fold_densities)
            }
            experiment_logs.append(log_entry)
            print(f"-> skab analizi bitti | Window: {w_size}, Alphabet: {a_size} | F1-ortalama: {log_entry['f1_mean']:.4f}")

    df_results = pd.DataFrame(experiment_logs)
    df_results.to_csv("automata_skab_param_results.csv", index=False)
    print("✅ skab sonuçları automata_skab_param_results dosyasına kaydedildi.")


def run_batadal_experiments(batadal_df: pd.DataFrame):
    """
    BATADAL veri seti için Zaman Sıralı (%60 Train, %20 Val, %20 Test)
    veri sızıntısız parametre analiz döngüsü.
    """
    target_col = "ATT_FLAG"
    sensor_cols = [col for col in batadal_df.columns if col not in ["DATETIME", target_col]]

    X = batadal_df[sensor_cols].values
    y = batadal_df[target_col].values

    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=1)
    X_train_pc1 = pca.fit_transform(X_train_scaled).flatten()
    X_val_pc1 = pca.transform(X_val_scaled).flatten()
    X_test_pc1 = pca.transform(X_test_scaled).flatten()

    experiment_logs = []
    print("\nbatadal veriseti için analiz başladı")

    for w_size in settings.PARAM_WINDOW_SIZES:
        for a_size in settings.PARAM_ALPHABET_SIZES:
            pipeline = AutomataPipeline(paa_window_size=w_size, pattern_window_size=w_size, alphabet_size=a_size)
            fit_res = pipeline.fit(X_train_pc1)

            val_predictions = pipeline.predict(X_val_pc1)
            test_predictions = pipeline.predict(X_test_pc1)

            # metrik bağlantı noktaları
            val_metrics_res = EvaluationMetrics.evaluate(y_val, val_predictions)
            test_metrics_res = EvaluationMetrics.evaluate(y_test, test_predictions)

            f1_val = val_metrics_res["f1_score"]
            f1_test = test_metrics_res["f1_score"]

            states_count = len(fit_res["states"])
            density = len(fit_res["transition_table"]) / (states_count ** 2 if states_count > 0 else 1)

            experiment_logs.append(
                {
                    "window_size": w_size,
                    "alphabet_size": a_size,
                    "f1_val": f1_val,
                    "f1_test": f1_test,
                    "state_count": states_count,
                    "transition_density": density
                }
            )
            print(f"-> batadal analizi bitti Window: {w_size}, Alphabet: {a_size} | Val F1: {f1_val:.4f}, Test F1: {f1_test:.4f}")

    df_results = pd.DataFrame(experiment_logs)
    df_results.to_csv("automata_batadal_param_results.csv", index=False)
    print("✅ batadal sonuçları automata_batadal_param_results.csv dosyasına kaydedildi.")