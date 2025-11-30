import tensorflow as tf
from tensorflow.keras import layers, models
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

# Parámetros
max_balls = 6
hand_feats = 4
max_features = hand_feats + max_balls*2 + 1  # 17
mask_value = -1.0  # mejor que 0 si normalizas a [0,1]
data_root = "../CSVs/60fps128"  # carpeta raíz

# Loader
def load_dataset(root):
    X_list, y_list = [], []
    label_map = {}
    next_label = 0
    max_len = 0
    csv_files = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    for path in csv_files:
        fname = os.path.basename(path)[:-4]
        tokens = fname.split("_")
        if len(tokens) < 2:
            continue  # necesitar al menos nballs y trickname
        nballs_str = tokens[0]
        try:
            nballs = int(nballs_str)
        except:
            continue
        # Detectar si último token es id numérico
        if len(tokens) > 2 and tokens[-1].isdigit():
            trick_tokens = tokens[1:-1]
        else:
            trick_tokens = tokens[1:]
        trickname = "_".join(trick_tokens).lower()
        if trickname == "":
            continue
        if trickname not in label_map:
            label_map[trickname] = next_label
            next_label += 1
        y = label_map[trickname]

        data = pd.read_csv(path, header=None).values
        # Construir secuencia con padding de columnas hasta 16
        seq = np.full((data.shape[0], max_features), mask_value, dtype=np.float32)
        # data tiene 4 + 2*nballs columnas reales
        real_cols = 4 + 2*nballs
        seq[:, :real_cols] = data[:, :real_cols]
        seq[:, -1] = nballs / max_balls  # última columna = ball_count normalizado

        # Normalización por secuencia (solo frames reales)
        # Identificar frames reales (por manos no en mask_value)
        real_mask = np.any(seq[:, :hand_feats] != mask_value, axis=1)
        real_frames = seq[real_mask, :real_cols]
        if real_frames.size == 0:
            continue

        # Min-max (sobre frames reales) evitando división por 0
        minv = real_frames.min(axis=0, keepdims=True)
        maxv = real_frames.max(axis=0, keepdims=True)
        rangev = maxv - minv
        rangev[rangev == 0] = 1.0
        scaled = (real_frames - minv) / rangev  # [0,1]

        # Insertar normalizado en sus columnas reales (comentado porque no es algo que Sergio dijo)
        seq[real_mask, :real_cols] = scaled #real_frames
        # Columnas vacías permanecen en mask_value (-1)
        X_list.append(seq)
        y_list.append(y)
        max_len = max(max_len, seq.shape[0])

    num_classes = next_label
    # Pad temporal
    X_padded = np.full((len(X_list), max_len, max_features), mask_value, dtype=np.float32)
    for i, seq in enumerate(X_list):
        X_padded[i, :seq.shape[0]] = seq
    y_arr = np.array(y_list, dtype=np.int32)
    return X_padded, y_arr, num_classes, label_map

if __name__ == "__main__":
    X, y, num_classes, label_map = load_dataset(data_root)
    print("Dataset:", X.shape, y.shape, "num_classes:", num_classes)
    
    # Parámetros CV
    K_FOLDS = 5
    EPOCHS = 50
    BATCH = 32
    
    os.makedirs("checkpoints", exist_ok=True)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    fold_f1_macro = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== FOLD {fold+1}/{K_FOLDS} ===")
        
        # Split train/val dentro del fold
        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        test_X, test_y = X[test_idx], y[test_idx]
        
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42+fold)
        train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))
        train_X, train_y = X_train_val[train_idx], y_train_val[train_idx]
        val_X, val_y = X_train_val[val_idx], y_train_val[val_idx]
        
        # Construir modelo (función para reutilizar)
        def build_model():
            inp = layers.Input(shape=(None, max_features), name="coords")  # 17 features
            x = layers.Masking(mask_value=mask_value)(inp)
            x = layers.Conv1D(128, 3, activation='relu')(x)
            x = layers.Conv1D(256, 5, padding='same', activation='relu', dilation_rate=3)(x)
            x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(64, activation='relu')(x)
            out = layers.Dense(num_classes, activation='softmax')(x)
            
            model = models.Model(inputs=inp, outputs=out)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        
        model = build_model()
        
        # Callbacks
        early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        checkpoint = ModelCheckpoint(
            f"checkpoints/fold_{fold+1}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
        
        # Entrenar
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=EPOCHS,
            batch_size=BATCH,
            callbacks=[early, checkpoint],
            verbose=1
        )
        
        # Evaluar en test del fold
        test_pred_probs = model.predict(test_X)
        test_pred = np.argmax(test_pred_probs, axis=1)
        acc = (test_pred == test_y).mean()
        f1_macro = f1_score(test_y, test_pred, average='macro')
        print(f"Fold {fold+1} Test Accuracy: {acc:.3f}")
        print(f"Fold {fold+1} F1-score (macro): {f1_macro:.3f}")
        fold_results.append(acc)
        fold_f1_macro.append(f1_macro)
        
        # Classification report
        idx_to_label = {v:k for k,v in label_map.items()}
        labels_sorted = [idx_to_label[i] for i in range(num_classes)]
        print(f"\n--- Classification Report Fold {fold+1} ---")
        print(classification_report(test_y, test_pred, target_names=labels_sorted, zero_division=0))
        
        # Matriz de confusión por fold
        cm = confusion_matrix(test_y, test_pred)
        plt.figure(figsize=(10,8))
        plt.imshow(cm, cmap='Blues')
        plt.title(f'Matriz de confusión - Fold {fold+1}')
        plt.xticks(range(num_classes), labels_sorted, rotation=90)
        plt.yticks(range(num_classes), labels_sorted)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_fold_{fold+1}.png")
        plt.close()

    # Resumen CV
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Fold accuracies: {fold_results}")
    print(f"Mean accuracy: {np.mean(fold_results):.3f} ± {np.std(fold_results):.3f}")
    print(f"Fold F1-macro: {fold_f1_macro}")
    print(f"Mean F1-macro: {np.mean(fold_f1_macro):.3f} ± {np.std(fold_f1_macro):.3f}")

    # Guardar resultados
    with open("cv_results.json", "w") as f:
        json.dump({
            "fold_accuracies": fold_results,
            "fold_f1_macro": fold_f1_macro,
            "mean_accuracy": float(np.mean(fold_results)),
            "std_accuracy": float(np.std(fold_results)),
            "mean_f1_macro": float(np.mean(fold_f1_macro)),
            "std_f1_macro": float(np.std(fold_f1_macro))
        }, f, indent=2)

    # Guardar label_map
    with open("label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
