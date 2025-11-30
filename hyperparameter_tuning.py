import tensorflow as tf
from tensorflow.keras import layers, models
import kerastuner as kt
import numpy as np
import json
# Importar load_dataset desde arquinew
import sys
sys.path.insert(0, '.')
from arquinew import load_dataset, max_features, mask_value

X, y, num_classes, label_map = load_dataset("../CSVs/60fps128")

print("X shape:", X.shape)  # debe ser (n_samples, max_len, 17)
print("Sample features:", X[0, 0, :])  # última columna debe ser nballs/6

# Split para tuning
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def build_model(hp):
    inp = layers.Input(shape=(None, max_features), name="coords")  # 17 features
    x = layers.Masking(mask_value=mask_value)(inp)
    
    # Hiperparámetros
    filters_base = hp.Choice('filters_base', [32, 64, 128])
    kernel_size = hp.Choice('kernel_size', [3, 5, 7])
    dilation = hp.Choice('dilation', [2, 3, 4])
    dense_units = hp.Choice('dense_units', [64, 128, 256])
    dropout_rate = hp.Choice('dropout', [0.3, 0.4, 0.5])
    lr = hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3])
    
    x = layers.Conv1D(filters_base, kernel_size, padding='same', activation='relu')(x)
    x = layers.Conv1D(filters_base*2, 5, padding='same', activation='relu', dilation_rate=dilation)(x)
    x = layers.Conv1D(filters_base*2, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(  # cambiar a RandomSearch (compatible TF 1.15)
    build_model,
    objective='val_acc',
    max_trials=30,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='juggling_random'
)

tuner.search(
    train_X, train_y,  # solo 1 input
    validation_data=(val_X, val_y),  # solo 1 input
    epochs=30,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
)

# Mejores hiperparámetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n=== BEST HYPERPARAMETERS ===")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Guardar
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_hps.values, f, indent=2)