import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from settings import settings
from automata.pipeline import AutomataPipeline
from evaluation.metrics import EvaluationMetrics


def skab_to_pc1(skab_df):
    exclude_cols = ["datetime", "anomaly", "changepoint", "source_file", "source_group"]
    sensor_cols = [col for col in skab_df.columns if col not in exclude_cols]

    X = skab_df[sensor_cols].values
    y = skab_df["anomaly"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).flatten()

    return pc1, y


def batadal_to_pc1(batadal_df):
    target_col = "ATT_FLAG"
    batadal_df[target_col] = batadal_df[target_col].replace(-999, 0)

    exclude_cols = ["DATETIME", target_col]
    sensor_cols = [col for col in batadal_df.columns if col not in exclude_cols]

    X = batadal_df[sensor_cols].values
    y = batadal_df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).flatten()

    return pc1, y


def run_cross_dataset_experiments(skab_df, batadal_df):
    skab_pc1, skab_y = skab_to_pc1(skab_df)
    batadal_pc1, batadal_y = batadal_to_pc1(batadal_df)

    experiments = [
        {
            "train_dataset": "SKAB",
            "test_dataset": "BATADAL",
            "train_series": skab_pc1,
            "test_series": batadal_pc1,
            "y_test": batadal_y
        },
        {
            "train_dataset": "BATADAL",
            "test_dataset": "SKAB",
            "train_series": batadal_pc1,
            "test_series": skab_pc1,
            "y_test": skab_y
        }
    ]

    results = []

    for exp in experiments:
        seed_f1_scores = []

        for seed in settings.RANDOM_SEEDS:
            np.random.seed(seed)

            pipeline = AutomataPipeline(
                paa_window_size=4,
                pattern_window_size=4,
                alphabet_size=3
            )

            pipeline.fit(exp["train_series"])
            predictions = pipeline.predict(exp["test_series"])

            metrics = EvaluationMetrics.evaluate(exp["y_test"], predictions)
            seed_f1_scores.append(metrics["f1_score"])

        results.append({
            "train_dataset": exp["train_dataset"],
            "test_dataset": exp["test_dataset"],
            "f1_mean": np.mean(seed_f1_scores),
            "f1_std": np.std(seed_f1_scores)
        })

        print(
            f"[CROSS] Train: {exp['train_dataset']} -> Test: {exp['test_dataset']} | "
            f"F1: {np.mean(seed_f1_scores):.4f} ± {np.std(seed_f1_scores):.4f}"
        )

    os.makedirs(settings.BASE_LOG_DIR, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(
        os.path.join(settings.BASE_LOG_DIR, "cross_dataset_results.csv"),
        index=False
    )

    print("Cross-dataset sonuçları logs/cross_dataset_results.csv dosyasına kaydedildi.")
    return df_results