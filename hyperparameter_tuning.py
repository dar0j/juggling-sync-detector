import tensorflow as tf
from tensorflow.keras import layers, models
import kerastuner as kt
import numpy as np
import json
# Importar load_dataset desde arquinew
import sys
sys.path.insert(0, '.')
from arquinew import load_dataset, max_features, mask_value

X, y, num_classes, label_map = load_dataset("../CSVs/60fps128", shuffle_balls=True)

print("X shape:", X.shape)  # debe ser (n_samples, max_len, 17)
print("Sample features:", X[0, 0, :])  # última columna debe ser nballs/6

# Split para tuning
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

class JugglingHyperModel(kt.HyperModel):
    def build(self, hp):
        inp = layers.Input(shape=(None, max_features), name="coords")
        x = layers.Masking(mask_value=mask_value)(inp)
        
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
    
    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice('batch_size', [16, 32, 64])
        return model.fit(*args, batch_size=batch_size, **kwargs)

hypermodel = JugglingHyperModel()
tuner = kt.Hyperband(
    hypermodel,
    objective='val_acc',
    max_epochs=30,
    factor=3,  # reduce épocas en 3x en cada ronda
    hyperband_iterations=2,
    directory='tuner_results',
    project_name='juggling_hyperband'
)

tuner.search(train_X, train_y, validation_data=(val_X, val_y))

# Resumen de búsqueda
tuner.results_summary(num_trials=5)  # top 5 trials

print("\n=== SEARCH SUMMARY ===")
print(f"Total trials: {len(tuner.oracle.trials)}")
print(f"Completed trials: {len([t for t in tuner.oracle.trials.values() if t.status == 'COMPLETED'])}")

# Obtener mejor modelo
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save("best_tuned_model.h5")
print("✅ Modelo guardado: best_tuned_model.h5")

# Evaluar en validación
val_loss, val_acc = best_model.evaluate(val_X, val_y, verbose=0)
print(f"Mejor modelo - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

# Obtener top 3 configuraciones
best_hps = tuner.get_best_hyperparameters(num_trials=3)
print("\n=== TOP 3 HYPERPARAMETERS ===")
for i, hps in enumerate(best_hps, 1):
    print(f"\nRank {i}:")
    for param, value in hps.values.items():
        print(f"  {param}: {value}")

# Guardar top 3
with open("best_hyperparameters.json", "w") as f:
    json.dump({
        f"rank_{i}": hps.values 
        for i, hps in enumerate(best_hps, 1)
    }, f, indent=2)
print("\n💾 Hiperparámetros guardados en best_hyperparameters.json")


# Resultados del mejor trial
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
if best_trial.score is not None:
    print(f"\nBest trial val_acc: {best_trial.score:.4f}")
else:
    print("\n⚠️  Mejor trial no completado correctamente")