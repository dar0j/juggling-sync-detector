import tensorflow as tf
from tensorflow.keras import layers, models
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Parámetros
max_balls = 6
hand_feats = 4
max_features = hand_feats + max_balls*2  # 16
mask_value = -1.0  # mejor que 0 si normalizas a [0,1]
data_root = "../CSVs/60fps128"  # carpeta raíz

# Loader
def load_dataset(root):
    X_list, y_list, ball_count_list = [], [], []
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

        # Insertar normalizado en sus columnas reales
        seq[real_mask, :real_cols] = scaled
        # Columnas vacías permanecen en mask_value (-1)
        X_list.append(seq)
        y_list.append(y)
        ball_count_list.append(nballs)
        max_len = max(max_len, seq.shape[0])

    num_classes = next_label
    # Pad temporal
    X_padded = np.full((len(X_list), max_len, max_features), mask_value, dtype=np.float32)
    for i, seq in enumerate(X_list):
        X_padded[i, :seq.shape[0]] = seq
    y_arr = np.array(y_list, dtype=np.int32)
    ball_counts = np.array(ball_count_list, dtype=np.float32).reshape(-1,1)
    return X_padded, y_arr, ball_counts, num_classes, label_map

X, y, ball_counts, num_classes, label_map = load_dataset(data_root)
print("Dataset:", X.shape, y.shape, ball_counts.shape, "num_classes:", num_classes)

# Split estratificado
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=43)  # ~15% val del total restante
remain_X, remain_y = X[train_idx], y[train_idx]
remain_bc = ball_counts[train_idx]
train_sub_idx, val_idx = next(sss2.split(remain_X, remain_y))
train_X, train_y, train_bc = remain_X[train_sub_idx], remain_y[train_sub_idx], remain_bc[train_sub_idx]
val_X, val_y, val_bc = remain_X[val_idx], remain_y[val_idx], remain_bc[val_idx]
test_X, test_y, test_bc = X[test_idx], y[test_idx], ball_counts[test_idx]

# Modelo
inp = layers.Input(shape=(None, max_features), name="coords") # Input: secuencia padded (T,16)
x = layers.Masking(mask_value=mask_value)(inp)
ball_count_inp = layers.Input(shape=(1,), name="ball_count")
seq_len = tf.shape(x)[1]
# bc = layers.Lambda(lambda t: tf.tile(t, [1, seq_len, 1]))(ball_count_inp) # Repetir scalar ball_count al largo dinámico (usa Lambda que tiró error de shape)
bc = layers.RepeatVector(seq_len)(ball_count_inp)
bc = tf.cast(bc, tf.float32)
x = layers.Concatenate()([x, bc])  # ahora features = 17

def multi_conv(t, filters): # Bloque multi-kernel
    c3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(t)
    c5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(t)
    c7 = layers.Conv1D(filters, 7, padding='same', activation='relu')(t)
    return layers.Concatenate()([c3, c5, c7])

x = multi_conv(x, 64)
x = layers.Conv1D(128, 5, padding='same', activation='relu', dilation_rate=2)(x)
x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
out = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=[inp, ball_count_inp], outputs=out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenamiento
EPOCHS = 50
BATCH = 32
history = model.fit(
    [train_X, train_bc], train_y,
    validation_data=([val_X, val_bc], val_y),
    epochs=EPOCHS,
    batch_size=BATCH
)

# Evaluación test
test_pred_probs = model.predict([test_X, test_bc])
test_pred = np.argmax(test_pred_probs, axis=1)
acc = (test_pred == test_y).mean()
print(f"Test accuracy: {acc:.3f}")

# Matriz de confusión
cm = confusion_matrix(test_y, test_pred)
idx_to_label = {v:k for k,v in label_map.items()}
labels_sorted = [idx_to_label[i] for i in range(num_classes)]
plt.figure(figsize=(10,8))
plt.imshow(cm, cmap='Blues')
plt.title('Matriz de confusión')
plt.xticks(range(num_classes), labels_sorted, rotation=90)
plt.yticks(range(num_classes), labels_sorted)
plt.colorbar()
plt.tight_layout()
plt.show()
