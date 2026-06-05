import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from settings import settings
from automata.pipeline import AutomataPipeline
from evaluation.metrics import EvaluationMetrics
from experiments.preprocessing import Utils


def save_detailed_logs(explanations: list, dataset_name: str, scenario_name: str, w_size: int, a_size: int, seed: int):
    """
    Her parametre seti, seed ve senaryo (original/noisy) için adım adım olasılıksal açıklamaları
    izole bir şekilde JSON ve CSV olarak kaydeder (VIII.A İsteri).
    """
    if not explanations:
        return

    # Klasör yolunu settings'ten alıp yapılandırıyoruz
    log_dir = os.path.join(settings.AUTOMATA_LOG_DIR, dataset_name, scenario_name)
    os.makedirs(log_dir, exist_ok=True)

    file_name = f"explain_w{w_size}_a{a_size}_seed{seed}"
    json_path = os.path.join(log_dir, f"{file_name}.json")
    csv_path = os.path.join(log_dir, f"{file_name}.csv")

    # JSON Formatında Kaydetme
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=4, ensure_ascii=False)

    # CSV (Tablo) Formatında Kaydetme
    df_explain = pd.DataFrame(explanations)
    columns_order = ["time_step", "state", "pattern", "status", "mapped_to",
                     "probability", "path_probability", "decision", "confidence", "distance"]
    available_columns = [col for col in columns_order if col in df_explain.columns]
    df_explain = df_explain[available_columns]
    df_explain.to_csv(csv_path, index=False)


def run_skab_experiments(skab_df: pd.DataFrame):
    """
    SKAB için parametre analizi, 5-Seed ve Gaussian Gürültü test döngüsü.
    """
    exclude_cols = ["datetime", "anomaly", "changepoint", "source_file", "source_group"]
    sensor_cols = [col for col in skab_df.columns if col not in exclude_cols]

    groups = skab_df["source_file"].values
    X = skab_df[sensor_cols].values
    y = skab_df["anomaly"].values

    gkf = GroupKFold(n_splits=5)
    experiment_logs = []

    print("\n[AUTOMATA-SKAB] Veriseti için analiz başladı...")

    for w_size in settings.PARAM_WINDOW_SIZES:
        for a_size in settings.PARAM_ALPHABET_SIZES:

            # Parametre kombinasyonu altındaki seed sonuçlarını biriktirmek için listeler
            seed_orig_means = []
            seed_noisy_means = []
            all_state_counts = []
            all_densities = []

            for seed in settings.RANDOM_SEEDS:
                np.random.seed(seed)  # Tekrarlanabilirlik kalkanı

                fold_orig_f1 = []
                fold_noisy_f1 = []

                last_fold_predictions_orig = None
                last_fold_predictions_noisy = None

                for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    pca = PCA(n_components=1)
                    X_train_pc1 = pca.fit_transform(X_train_scaled).flatten()
                    X_test_pc1 = pca.transform(X_test_scaled).flatten()

                    # VII. Senaryo uyarınca gürültülü test verisini türetiyoruz
                    X_test_pc1_noisy = Utils.add_gaussian_noise(X_test_pc1, settings.AUTOMATA_SKAB_NOISE_LEVEL)

                    pipeline = AutomataPipeline(paa_window_size=w_size, pattern_window_size=w_size, alphabet_size=a_size)
                    fit_res = pipeline.fit(X_train_pc1)

                    # 1. Orijinal Test Kümesi Tahmini
                    pipeline.explainer.reset_path()
                    predictions_orig = pipeline.predict(X_test_pc1)
                    last_fold_predictions_orig = predictions_orig
                    f1_orig = EvaluationMetrics.evaluate(y_test, predictions_orig)["f1_score"]
                    fold_orig_f1.append(f1_orig)

                    # 2. Gürültülü Test Kümesi Tahmini
                    pipeline.explainer.reset_path()
                    predictions_noisy = pipeline.predict(X_test_pc1_noisy)
                    last_fold_predictions_noisy = predictions_noisy
                    f1_noisy = EvaluationMetrics.evaluate(y_test, predictions_noisy)["f1_score"]
                    fold_noisy_f1.append(f1_noisy)

                    # Otomata Yapısal İstatistikleri
                    states_count = len(fit_res["states"])
                    density = len(fit_res["transition_table"]) / (states_count ** 2 if states_count > 0 else 1)
                    all_state_counts.append(states_count)
                    all_densities.append(density)

                # Her seed bittiğinde o seed'e ait fold ortalamalarını ana listeye ekle
                seed_orig_means.append(np.mean(fold_orig_f1))
                seed_noisy_means.append(np.mean(fold_noisy_f1))

                # Son fold'un detaylı adımlarını diske izole et
                save_detailed_logs(last_fold_predictions_orig, "SKAB", "original", w_size, a_size, seed)
                save_detailed_logs(last_fold_predictions_noisy, "SKAB", "noisy", w_size, a_size, seed)

            # 5 Seed'in genel istatistiksel özeti hesaplanır
            log_entry = {
                "window_size": w_size,
                "alphabet_size": a_size,
                "f1_orig_mean": np.mean(seed_orig_means),
                "f1_orig_std": np.std(seed_orig_means),
                "f1_noisy_mean": np.mean(seed_noisy_means),
                "f1_noisy_std": np.std(seed_noisy_means),
                "state_count_mean": np.mean(all_state_counts),
                "transition_density_mean": np.mean(all_densities)
            }
            experiment_logs.append(log_entry)

            # Konsolda her iki senaryonun da performansı net bir biçimde listelenir
            print(
                f"-> SKAB Bitti | Window: {w_size}, Alphabet: {a_size} | "
                f"Orig F1-Mean: {log_entry['f1_orig_mean']:.4f} (±{log_entry['f1_orig_std']:.3f}) | "
                f"Noisy F1-Mean: {log_entry['f1_noisy_mean']:.4f} (±{log_entry['f1_noisy_std']:.3f})"
                )

    # Özet sonuçları merkezi kayıt klasörüne basma
    os.makedirs(settings.AUTOMATA_LOG_DIR, exist_ok=True)
    df_results = pd.DataFrame(experiment_logs)
    df_results.to_csv(os.path.join(settings.AUTOMATA_LOG_DIR, "automata_skab_param_results.csv"), index=False)
    print("✅ SKAB genel özet sonuçları ve senaryo logları başarıyla kaydedildi.")


def run_batadal_experiments(batadal_df: pd.DataFrame):
    """
    BATADAL için parametre analizi, 5-Seed ve Gaussian Gürültü test döngüsü.
    """
    target_col = "ATT_FLAG"
    sensor_cols = [col for col in batadal_df.columns if col not in ["DATETIME", target_col]]

    X = batadal_df[sensor_cols].values
    batadal_df[target_col] = batadal_df[target_col].replace(-999, 0)
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

    # Gürültülü test ve validasyon verilerini hazırlıyoruz
    X_val_pc1_noisy = Utils.add_gaussian_noise(X_val_pc1, settings.AUTOMATA_BATADAL_NOISE_LEVEL)
    X_test_pc1_noisy = Utils.add_gaussian_noise(X_test_pc1, settings.AUTOMATA_BATADAL_NOISE_LEVEL)

    experiment_logs = []
    print("\n[AUTOMATA-BATADAL] Veriseti için analiz başladı...")

    for w_size in settings.PARAM_WINDOW_SIZES:
        for a_size in settings.PARAM_ALPHABET_SIZES:

            seed_val_orig, seed_test_orig = [], []
            seed_val_noisy, seed_test_noisy = [], []
            all_state_counts = []
            all_densities = []

            for seed in settings.RANDOM_SEEDS:
                np.random.seed(seed)

                pipeline = AutomataPipeline(paa_window_size=w_size, pattern_window_size=w_size, alphabet_size=a_size)
                fit_res = pipeline.fit(X_train_pc1)

                # 1. Orijinal Veri Tahminleri
                pipeline.explainer.reset_path()
                val_preds_orig = pipeline.predict(X_val_pc1)
                pipeline.explainer.reset_path()
                test_preds_orig = pipeline.predict(X_test_pc1)

                # 2. Gürültülü Veri Tahminleri
                pipeline.explainer.reset_path()
                val_preds_noisy = pipeline.predict(X_val_pc1_noisy)
                pipeline.explainer.reset_path()
                test_preds_noisy = pipeline.predict(X_test_pc1_noisy)

                # Skorların Hesaplanması
                seed_val_orig.append(EvaluationMetrics.evaluate(y_val, val_preds_orig)["f1_score"])
                seed_test_orig.append(EvaluationMetrics.evaluate(y_test, test_preds_orig)["f1_score"])
                seed_val_noisy.append(EvaluationMetrics.evaluate(y_val, val_preds_noisy)["f1_score"])
                seed_test_noisy.append(EvaluationMetrics.evaluate(y_test, test_preds_noisy)["f1_score"])

                # Yapısal Veri Takibi
                states_count = len(fit_res["states"])
                density = len(fit_res["transition_table"]) / (states_count ** 2 if states_count > 0 else 1)
                all_state_counts.append(states_count)
                all_densities.append(density)

                # Adım Günlüklerini Kaydetme
                save_detailed_logs(test_preds_orig, "BATADAL", "original", w_size, a_size, seed)
                save_detailed_logs(test_preds_noisy, "BATADAL", "noisy", w_size, a_size, seed)

            log_entry = {
                "window_size": w_size, "alphabet_size": a_size,
                "f1_val_orig_mean": np.mean(seed_val_orig), "f1_test_orig_mean": np.mean(seed_test_orig),
                "f1_test_orig_std": np.std(seed_test_orig),
                "f1_val_noisy_mean": np.mean(seed_val_noisy), "f1_test_noisy_mean": np.mean(seed_test_noisy),
                "f1_test_noisy_std": np.std(seed_test_noisy),
                "state_count_mean": np.mean(all_state_counts), "transition_density_mean": np.mean(all_densities)
            }
            experiment_logs.append(log_entry)

            print(
                f"-> BATADAL Bitti | Window: {w_size}, Alphabet: {a_size} | "
                f"Test F1 Orig: {log_entry['f1_test_orig_mean']:.4f} (±{log_entry['f1_test_orig_std']:.3f}) | "
                f"Test F1 Noisy: {log_entry['f1_test_noisy_mean']:.4f} (±{log_entry['f1_test_noisy_std']:.3f})"
                )

    # Özet sonuçları diskteki merkezi klasörle buluşturma
    os.makedirs(settings.AUTOMATA_LOG_DIR, exist_ok=True)
    df_results = pd.DataFrame(experiment_logs)
    df_results.to_csv(os.path.join(settings.AUTOMATA_LOG_DIR, "automata_batadal_param_results.csv"), index=False)
    print("✅ BATADAL genel özet sonuçları ve senaryo logları başarıyla kaydedildi.")