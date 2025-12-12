import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
import json
from arquinew import load_dataset

# Cargar datos
data_root = "../CSVs/60fps128"
X, y, num_classes, label_map = load_dataset(data_root)
print(f"Dataset: {X.shape}, {num_classes} clases")

# Configuración CV
K_FOLDS = 5
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Estrategias dummy
strategies = ['uniform', 'stratified', 'most_frequent']
results = {strat: {'accuracy': [], 'f1_macro': [], 'balanced_accuracy': []} for strat in strategies}

idx_to_label = {v:k for k,v in label_map.items()}
labels_sorted = [idx_to_label[i] for i in range(num_classes)]

first_fold_preds = {}  # NUEVO: almacenar predicciones del fold 1

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== FOLD {fold+1}/{K_FOLDS} ===")
    train_y, test_y = y[train_idx], y[test_idx]
    
    for strategy in strategies:
        clf = DummyClassifier(strategy=strategy, random_state=42)
        clf.fit(np.zeros((len(train_y), 1)), train_y)  # dummy no usa features
        pred_y = clf.predict(np.zeros((len(test_y), 1)))
        
        # Guardar predicciones del fold 1 para reporte detallado
        if fold == 0 and strategy == 'stratified':
            first_fold_preds['test_y'] = test_y
            first_fold_preds['pred_y'] = pred_y
        
        acc = accuracy_score(test_y, pred_y)
        f1 = f1_score(test_y, pred_y, average='macro', zero_division='warn')
        balanced_acc = balanced_accuracy_score(test_y, pred_y)
        
        results[strategy]['accuracy'].append(acc)
        results[strategy]['f1_macro'].append(f1)
        results[strategy]['balanced_accuracy'].append(balanced_acc)
        
        print(f"{strategy:15} - Acc: {acc:.3f}, F1-macro: {f1:.3f}, Balanced-Acc: {balanced_acc:.3f}")

# Resumen
print("\n=== BASELINE RESULTS (5-Fold CV) ===")
for strategy in strategies:
    print(f"\n{strategy.upper()}:")
    for metric, values in results[strategy].items():
        print(f"  {metric:20}: {np.mean(values):.3f} ± {np.std(values):.3f}")

# Guardar
with open("dummy_baseline_results.json", "w") as f:
    summary = {strat: {metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                       for metric, vals in metrics.items()}
               for strat, metrics in results.items()}
    json.dump(summary, f, indent=2)

# Reporte detallado usando predicciones guardadas
print("\n=== CLASSIFICATION REPORT (Stratified Dummy, Fold 1) ===")
print(classification_report(
    first_fold_preds['test_y'], 
    first_fold_preds['pred_y'], 
    target_names=labels_sorted, 
    zero_division=0
))