import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from settings import settings
from experiments.preprocessing import DeepLearningPreprocessor , Utils
from experiments.models import LSTMModel, GRUModel


def configure_seed(seed):
    """Deneylerin tekrarlanabilirliği için random seedler en baştan sabitlenir"""
    # sabit kararlılık için önceki oturum temizlenir
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_dl_detailed_logs(y_true, y_pred_probs, dataset_name: str, scenario_name: str, model_type: str, seed: int):
    """
    test kümesindeki adım adım tahminleri kaydeder
    """
    # klasör yolu seneryo ve verinin ismine göre dinamik belirlenir
    log_dir = os.path.join(settings.DEEP_LEARNING_LOG_DIR, dataset_name, scenario_name)
    os.makedirs(log_dir, exist_ok=True)

    file_name = f"{model_type.lower()}_seed{seed}"
    detailed_data = []

    for i in range(len(y_true)):
        prob = float(y_pred_probs[i])
        detailed_data.append(
            {
                "time_step": i + 1,
                "actual_label": int(y_true[i]),
                "prediction_probability": prob,
                "decision": "anomaly" if prob > 0.5 else "normal"
            }
        )

    json_path = os.path.join(log_dir, f"{file_name}.json")
    csv_path = os.path.join(log_dir, f"{file_name}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=4, ensure_ascii=False)

    pd.DataFrame(detailed_data).to_csv(csv_path, index=False)


def save_only_noisy_anomalies_cumulative(all_seeds_noisy_anomalies: list, dataset_name: str, model_type: str):
    """
    Mevcut loglama yapısına dokunmadan, tüm seed döngülerinden toplanan ve karar mekanizması
    'anomaly' olan gürültülü kayıtları içeren toptan bir kümülatif dosya oluşturur.
    """
    if not all_seeds_noisy_anomalies:
        return

    log_dir = os.path.join(settings.DEEP_LEARNING_LOG_DIR, dataset_name, "noisy")
    os.makedirs(log_dir, exist_ok=True)

    file_name = f"{model_type.lower()}_all_seeds_anomalies"
    json_path = os.path.join(log_dir, f"{file_name}.json")
    csv_path = os.path.join(log_dir, f"{file_name}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_seeds_noisy_anomalies, f, indent=4, ensure_ascii=False)

    df_anomalies = pd.DataFrame(all_seeds_noisy_anomalies)
    columns_order = ["seed", "time_step", "actual_label", "prediction_probability", "decision"]
    df_anomalies = df_anomalies[columns_order]
    df_anomalies.to_csv(csv_path, index=False)


def run_dl_skab_experiments(skab_df: pd.DataFrame, model_type="LSTM"):
    """pca kullanan skab verisi için deep learning modeli çalıştırıcı"""
    exclude_cols = ["datetime", "anomaly", "changepoint", "source_file", "source_group"]
    groups = skab_df["source_file"].values
    gkf = GroupKFold(n_splits=5)

    seed_results = []
    cumulative_noisy_anomalies = []

    for seed in settings.RANDOM_SEEDS:
        configure_seed(seed)

        fold_orig_f1 = []
        fold_noisy_f1 = []

        last_fold_y_true = None
        last_fold_preds_orig = None
        last_fold_preds_noisy = None

        print(f"\nSKAB derin öğrenme modeli: {model_type}  Seed: {seed} başladı...")

        for fold, (train_idx, test_idx) in enumerate(gkf.split(skab_df, groups=groups)):
            train_df = skab_df.iloc[train_idx]
            test_df = skab_df.iloc[test_idx]

            preprocessor = DeepLearningPreprocessor(n_components=1)
            preprocessor.fit(train_df, exclude_cols)

            X_train_raw, y_train_raw = preprocessor.transform(train_df, exclude_cols)
            X_test_raw, y_test_raw = preprocessor.transform(test_df, exclude_cols)

            # test verisinin gürültü eklenmiş hali üretilir
            X_test_raw_noisy = Utils.add_gaussian_noise(X_test_raw, settings.DEEP_LEARNING_SKAB_NOISE_LEVEL)

            window_size = 4
            X_train_3d, y_train = DeepLearningPreprocessor.create_3d_windows(X_train_raw, y_train_raw, window_size)
            X_test_3d_orig, y_test = DeepLearningPreprocessor.create_3d_windows(X_test_raw, y_test_raw, window_size)
            X_test_3d_noisy, _ = DeepLearningPreprocessor.create_3d_windows(X_test_raw_noisy, y_test_raw, window_size)

            model = LSTMModel(units=16) if model_type == "LSTM" else GRUModel(units=16)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )

            model.fit(
                X_train_3d, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )

            # orjinal veri tahmini
            preds_orig = model.predict(X_test_3d_orig, verbose=0).flatten()
            binary_preds_orig = (preds_orig > 0.5).astype(int)
            fold_orig_f1.append(f1_score(y_test, binary_preds_orig, average='macro', zero_division=0))

            # gürültülü veri tahmini
            preds_noisy = model.predict(X_test_3d_noisy, verbose=0).flatten()
            binary_preds_noisy = (preds_noisy > 0.5).astype(int)
            fold_noisy_f1.append(f1_score(y_test, binary_preds_noisy, average='macro', zero_division=0))

            last_fold_y_true = y_test
            last_fold_preds_orig = preds_orig
            last_fold_preds_noisy = preds_noisy

        # Detaylı adım ve log kayıtları
        save_dl_detailed_logs(last_fold_y_true, last_fold_preds_orig, "SKAB", "original", model_type, seed)
        save_dl_detailed_logs(last_fold_y_true, last_fold_preds_noisy, "SKAB", "noisy", model_type, seed)

        for i in range(len(last_fold_y_true)):
            prob = float(last_fold_preds_noisy[i])
            if prob > 0.5:
                cumulative_noisy_anomalies.append({
                    "seed": seed,
                    "time_step": i + 1,
                    "actual_label": int(last_fold_y_true[i]),
                    "prediction_probability": prob,
                    "decision": "anomaly"
                })

        seed_results.append(
            {
                "seed": seed,
                "f1_orig_mean": np.mean(fold_orig_f1),
                "f1_orig_std": np.std(fold_orig_f1),
                "f1_noisy_mean": np.mean(fold_noisy_f1),
                "f1_noisy_std": np.std(fold_noisy_f1)
            }
        )
        print(f"Seed: {seed} Bitti orjinal f1: {np.mean(fold_orig_f1):.4f}  gürültülü f1: {np.mean(fold_noisy_f1):.4f}")

    save_only_noisy_anomalies_cumulative(cumulative_noisy_anomalies, "SKAB", model_type)

    # merkezi özet dosya kaydı
    os.makedirs(settings.DEEP_LEARNING_LOG_DIR, exist_ok=True)
    df_results = pd.DataFrame(seed_results)

    summary_file_name = f"dl_skab_{model_type.lower()}_param_results.csv"
    summary_path = os.path.join(settings.DEEP_LEARNING_LOG_DIR, summary_file_name)
    df_results.to_csv(summary_path, index=False)

    return df_results


def run_dl_batadal_experiments(batadal_df: pd.DataFrame, model_type="LSTM"):
    """batadal df için pca kullanan deep learning model çalıştırıcı"""
    target_col = "ATT_FLAG"
    batadal_df[target_col] = batadal_df[target_col].replace(-999, 0)
    exclude_cols = ["DATETIME", target_col]

    n = len(batadal_df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = batadal_df.iloc[:train_end]
    val_df = batadal_df.iloc[train_end:val_end]
    test_df = batadal_df.iloc[val_end - 4:]

    seed_results = []
    cumulative_noisy_anomalies = []

    for seed in settings.RANDOM_SEEDS:
        configure_seed(seed)

        preprocessor = DeepLearningPreprocessor(n_components=1)
        preprocessor.fit(train_df, exclude_cols)

        X_train_raw, y_train_raw = preprocessor.transform(train_df, exclude_cols)
        X_val_raw, y_val_raw = preprocessor.transform(val_df, exclude_cols)
        X_test_raw, y_test_raw = preprocessor.transform(test_df, exclude_cols)

        # gaussian noise ekleme
        X_val_raw_noisy = Utils.add_gaussian_noise(X_val_raw, settings.DEEP_LEARNING_BATADAL_NOISE_LEVEL)
        X_test_raw_noisy = Utils.add_gaussian_noise(X_test_raw, settings.DEEP_LEARNING_BATADAL_NOISE_LEVEL)

        window_size = 4
        X_train_3d, y_train = DeepLearningPreprocessor.create_3d_windows(X_train_raw, y_train_raw, window_size)

        # orjinal veriler
        X_val_3d_orig, y_val = DeepLearningPreprocessor.create_3d_windows(X_val_raw, y_val_raw, window_size)
        X_test_3d_orig, y_test = DeepLearningPreprocessor.create_3d_windows(X_test_raw, y_test_raw, window_size)

        # gürültü eklenmiş veriler
        X_val_3d_noisy, _ = DeepLearningPreprocessor.create_3d_windows(X_val_raw_noisy, y_val_raw, window_size)
        X_test_3d_noisy, _ = DeepLearningPreprocessor.create_3d_windows(X_test_raw_noisy, y_test_raw, window_size)

        model = LSTMModel(units=16) if model_type == "LSTM" else GRUModel(units=16)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        # anomali verilerine ağırlık eklenir bu sebeple f1 skorunun sabit çıkması engellenir
        neg, pos = np.bincount(y_train.astype(int))
        total = neg + pos
        class_weight = {
            0: (1.0 / neg) * (total / 2.0),
            1: (1.0 / pos) * (total / 2.0)
        }

        # orjinal temiz veriyle eğitim bu sayede veri sızıntısı engellenir
        model.fit(
            X_train_3d, y_train,
            validation_data=(X_val_3d_orig, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0,
            class_weight = class_weight,
        )

        # orjinal küme tahmini ve skorlanması
        preds_orig = model.predict(X_test_3d_orig, verbose=0).flatten()
        binary_preds_orig = (preds_orig > 0.5).astype(int)
        f1_orig = f1_score(y_test, binary_preds_orig, average='macro', zero_division=0)

        # 2. gürültülü küme tahmini ve skorlanması
        preds_noisy = model.predict(X_test_3d_noisy, verbose=0).flatten()
        binary_preds_noisy = (preds_noisy > 0.5).astype(int)
        f1_noisy = f1_score(y_test, binary_preds_noisy, average='macro', zero_division=0)

        # detaylı adım log kayıtları
        save_dl_detailed_logs(y_test, preds_orig, "BATADAL", "original", model_type, seed)
        save_dl_detailed_logs(y_test, preds_noisy, "BATADAL", "noisy", model_type, seed)

        for i in range(len(y_test)):
            prob = float(preds_noisy[i])
            if prob > 0.5:
                cumulative_noisy_anomalies.append({
                    "seed": seed,
                    "time_step": i + 1,
                    "actual_label": int(y_test[i]),
                    "prediction_probability": prob,
                    "decision": "anomaly"
                })

        seed_results.append(
            {
                "seed": seed,
                "test_f1_original": f1_orig,
                "test_f1_noisy": f1_noisy
            }
        )
        print(f"BATADAL verisi derin öğrenme modeli: {model_type} | seed: {seed} | orjinal test f1: {f1_orig:.4f} | gürültülü test f1: {f1_noisy:.4f}")

    save_only_noisy_anomalies_cumulative(cumulative_noisy_anomalies, "BATADAL", model_type)

    # merkezi özet dosya kaydı
    os.makedirs(settings.DEEP_LEARNING_LOG_DIR, exist_ok=True)
    df_results = pd.DataFrame(seed_results)

    summary_file_name = f"dl_batadal_{model_type.lower()}_param_results.csv"
    summary_path = os.path.join(settings.DEEP_LEARNING_LOG_DIR, summary_file_name)
    df_results.to_csv(summary_path, index=False)

    return df_results