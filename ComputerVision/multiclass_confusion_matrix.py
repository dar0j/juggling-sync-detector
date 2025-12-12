#!/usr/bin/env python3
"""
multiclass_confusion_matrix.py
Genera matriz de confusión multiclase comparando expected vs largest_pattern
y calcula accuracy, macro F1-score y balanced accuracy.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_label_map(label_map_path='../label_map.json'):
    """Carga el label_map.json para obtener las 27 clases oficiales"""
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    # Invertir para obtener {index: class_name}
    idx_to_class = {v: k for k, v in label_map.items()}
    classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    return classes, label_map


def load_and_merge_results(file1, file2):
    """Carga y fusiona dos archivos JSON de resultados"""
    with open(file1, 'r') as f:
        data1 = json.load(f)
    
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # Fusionar
    merged = {**data1, **data2}
    return merged


def build_classification_data(results, valid_classes):
    """
    Construye listas de clases verdaderas y predichas.
    
    Lógica:
    - Clase verdadera = expected (extraída del filename)
    - Clase predicha = largest_pattern
    - Si largest_pattern no está en valid_classes, se marca como "unknown"
    """
    y_true = []
    y_pred = []
    filenames = []
    unknown_predictions = []
    
    for filename, data in results.items():
        true_class = data['expected']
        pred_class = data['largest_pattern']
        
        # Solo incluir si true_class está en valid_classes
        if true_class not in valid_classes:
            print(f"  [WARNING] Clase verdadera '{true_class}' no está en label_map (archivo: {filename})")
            continue
        
        # Si pred_class no está en valid_classes, contar como error pero no crear clase nueva
        if pred_class not in valid_classes:
            unknown_predictions.append((filename, true_class, pred_class))
            # Marcar como predicción especial (opcional: usar None o la clase verdadera para contar como FN)
            pred_class = None  # Esto contará como FN para la clase verdadera
        
        y_true.append(true_class)
        y_pred.append(pred_class)
        filenames.append(filename)
    
    if unknown_predictions:
        print(f"\n  [INFO] {len(unknown_predictions)} predicciones fuera de las 27 clases:")
        for fn, true_c, pred_c in unknown_predictions[:5]:  # Mostrar solo primeras 5
            print(f"    - {fn}: expected={true_c}, predicted={pred_c}")
        if len(unknown_predictions) > 5:
            print(f"    ... y {len(unknown_predictions) - 5} más")
    
    return y_true, y_pred, filenames


def calculate_multiclass_metrics(y_true, y_pred, classes):
    """
    Calcula métricas multiclase detalladas usando SOLO las clases válidas.
    
    Para cada clase:
    - TP: predicciones correctas de esa clase
    - FP: predicciones incorrectas como esa clase
    - FN: instancias de esa clase predichas como otra (o None)
    - TN: instancias de otras clases predichas correctamente como otras
    """
    metrics = {}
    
    for cls in classes:
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))  # incluye yp == None
        tn = sum((yt != cls and yp != cls) for yt, yp in zip(y_true, y_pred))
        
        # Métricas por clase
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cls] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support': tp + fn  # número real de instancias de esta clase
        }
    
    return metrics


def calculate_global_metrics_all_samples(y_true, y_pred, classes):
    """
    Calcula métricas globales sobre TODAS las muestras.
    Las predicciones None se consideran como una clase incorrecta especial.
    """
    # Reemplazar None con una clase dummy para cálculos
    y_pred_filled = [yp if yp is not None else '<PREDICTION_ERROR>' for yp in y_pred]
    
    # Accuracy simple: correctas / total
    accuracy_all = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    
    # Para F1 y Balanced Accuracy: calcular manualmente por clase
    # porque sklearn no maneja bien clases con 0 samples
    
    f1_scores = []
    recalls = []
    
    for cls in classes:
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores.append(f1)
        recalls.append(recall)
    
    # Macro F1: promedio de F1 de todas las clases
    macro_f1_all = np.mean(f1_scores)
    
    # Balanced Accuracy: promedio de recalls de todas las clases
    balanced_acc_all = np.mean(recalls)
    
    return {
        'accuracy': accuracy_all,
        'macro_f1': macro_f1_all,
        'balanced_accuracy': balanced_acc_all
    }


def plot_confusion_matrix(y_true, y_pred, classes, output_path='confusion_matrix.png'):
    """Genera y guarda matriz de confusión (solo 27 clases válidas)"""
    # Reemplazar None con clase especial para visualización
    y_pred_vis = [yp if yp is not None else '<unknown>' for yp in y_pred]
    
    # Agregar <unknown> a labels solo para visualización si hay casos
    labels_vis = classes + (['<unknown>'] if '<unknown>' in y_pred_vis else [])
    
    cm = confusion_matrix(y_true, y_pred_vis, labels=labels_vis)
    
    # Si hay muchas clases, hacer figura más grande
    n_classes = len(labels_vis)
    figsize = (max(14, n_classes * 0.6), max(12, n_classes * 0.5))
    
    plt.figure(figsize=figsize)
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_vis, yticklabels=labels_vis,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Siteswap Classification (27 Classes)', fontsize=14, pad=20)
    plt.xlabel('Predicted Pattern (largest_pattern)', fontsize=12)
    plt.ylabel('True Pattern (expected)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Matriz de confusión guardada en: {output_path}")
    plt.close()
    
    # También guardar versión normalizada
    plt.figure(figsize=figsize)
    
    # Normalizar fila por fila, manejando divisiones por cero
    row_sums = cm.sum(axis=1, keepdims=True)
    # Reemplazar 0 por 1 para evitar división por cero (filas sin muestras quedarán en 0)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm.astype('float') / row_sums_safe
    # Asegurar que filas sin muestras queden en 0
    cm_normalized = np.where(row_sums == 0, 0, cm_normalized)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels_vis, yticklabels=labels_vis,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    
    plt.title('Normalized Confusion Matrix - Siteswap Classification (27 Classes)', fontsize=14, pad=20)
    plt.xlabel('Predicted Pattern (largest_pattern)', fontsize=12)
    plt.ylabel('True Pattern (expected)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_normalized.png'), dpi=300, bbox_inches='tight')
    print(f"Matriz normalizada guardada en: {output_path.replace('.png', '_normalized.png')}")
    plt.close()


def main(file1, file2, label_map_path='../label_map.json', output_dir='classification_results'):
    """Pipeline principal"""
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*70)
    print("ANÁLISIS DE CLASIFICACIÓN MULTICLASE - SITESWAP DETECTION")
    print("="*70)
    
    # 1. Cargar clases válidas desde label_map.json
    print("\n1. Cargando label_map.json...")
    classes, label_map = load_label_map(label_map_path)
    print(f"   Total de clases válidas: {len(classes)}")
    print(f"   Clases: {classes[:5]}... (mostrando primeras 5)")
    
    # 2. Cargar datos
    print("\n2. Cargando archivos JSON...")
    results = load_and_merge_results(file1, file2)
    print(f"   Total de muestras: {len(results)}")
    
    # 3. Construir datos de clasificación (solo clases válidas)
    print("\n3. Extrayendo clases verdaderas y predichas...")
    y_true, y_pred, filenames = build_classification_data(results, set(classes))
    print(f"   Muestras válidas: {len(y_true)}")
    
    # 4. Calcular métricas por clase sobre TODAS las muestras (incluyendo None)
    print("\n4. Calculando métricas por clase (TODAS las muestras)...")
    class_metrics_all = calculate_multiclass_metrics(y_true, y_pred, classes)
    
    # 5. Métricas globales sobre TODAS las muestras (incluyendo None como errores)
    print("\n5. Calculando métricas globales sobre TODAS las muestras...")
    metrics_all = calculate_global_metrics_all_samples(y_true, y_pred, classes)
    accuracy_all = metrics_all['accuracy']
    macro_f1_all = metrics_all['macro_f1']
    balanced_acc_all = metrics_all['balanced_accuracy']
    
    # 6. Métricas por clase solo sobre predicciones válidas (sin None)
    print("\n6. Calculando métricas por clase (predicciones válidas)...")
    y_true_valid = [yt for yt, yp in zip(y_true, y_pred) if yp is not None]
    y_pred_valid = [yp for yp in y_pred if yp is not None]
    
    if len(y_true_valid) > 0:
        class_metrics_valid = calculate_multiclass_metrics(y_true_valid, y_pred_valid, classes)
        accuracy_valid = accuracy_score(y_true_valid, y_pred_valid)
        macro_f1_valid = f1_score(y_true_valid, y_pred_valid, labels=classes, average='macro', zero_division=0)
        balanced_acc_valid = balanced_accuracy_score(y_true_valid, y_pred_valid)
    else:
        class_metrics_valid = {cls: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 
                                     'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'Support': 0} 
                              for cls in classes}
        accuracy_valid = 0.0
        macro_f1_valid = 0.0
        balanced_acc_valid = 0.0
    
    # 7. Guardar métricas por clase en CSV (ambas versiones)
    print("\n7. Guardando resultados...")
    
    # CSV de todas las muestras
    df_metrics_all = pd.DataFrame(class_metrics_all).T
    df_metrics_all = df_metrics_all.sort_values('Support', ascending=False)
    csv_path_all = output_path / 'class_metrics_all_samples.csv'
    df_metrics_all.to_csv(csv_path_all)
    print(f"   Métricas por clase (TODAS) guardadas en: {csv_path_all}")
    
    # CSV de predicciones válidas
    df_metrics_valid = pd.DataFrame(class_metrics_valid).T
    df_metrics_valid = df_metrics_valid.sort_values('Support', ascending=False)
    csv_path_valid = output_path / 'class_metrics_valid_only.csv'
    df_metrics_valid.to_csv(csv_path_valid)
    print(f"   Métricas por clase (VÁLIDAS) guardadas en: {csv_path_valid}")
    
    # 8. Generar matriz de confusión
    print("\n8. Generando matrices de confusión...")
    plot_confusion_matrix(y_true, y_pred, classes, 
                         output_path=str(output_path / 'confusion_matrix.png'))
    
    # 9. Guardar reporte completo
    report_path = output_path / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE CLASIFICACIÓN MULTICLASE - SITESWAP DETECTION\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total de muestras válidas: {len(y_true)}\n")
        f.write(f"Total de clases (label_map): {len(classes)}\n")
        f.write(f"Predicciones fuera de label_map: {sum(yp is None for yp in y_pred)}\n\n")
        
        f.write("="*70 + "\n")
        f.write("SECCIÓN 1: MÉTRICAS SOBRE TODAS LAS MUESTRAS\n")
        f.write("="*70 + "\n\n")
        
        f.write("-"*70 + "\n")
        f.write("MÉTRICAS GLOBALES - TODAS LAS MUESTRAS (comparable con arquinew.py)\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:              {accuracy_all:.4f} ({accuracy_all*100:.2f}%)\n")
        f.write(f"Macro F1-Score:        {macro_f1_all:.4f}\n")
        f.write(f"Balanced Accuracy:     {balanced_acc_all:.4f} ({balanced_acc_all*100:.2f}%)\n\n")
        
        f.write("-"*70 + "\n")
        f.write("MÉTRICAS POR CLASE - TODAS LAS MUESTRAS\n")
        f.write("-"*70 + "\n\n")
        f.write(df_metrics_all.to_string())
        f.write("\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CLASES CON MEJOR F1-SCORE - TODAS LAS MUESTRAS (Top 10)\n")
        f.write("-"*70 + "\n\n")
        best_f1_all = df_metrics_all.nlargest(10, 'F1')
        f.write(best_f1_all[['F1', 'Precision', 'Recall', 'Support']].to_string())
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("SECCIÓN 2: MÉTRICAS SOBRE PREDICCIONES VÁLIDAS (sin <unknown>)\n")
        f.write("="*70 + "\n\n")
        
        f.write("-"*70 + "\n")
        f.write("MÉTRICAS GLOBALES - SOLO PREDICCIONES VÁLIDAS\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:              {accuracy_valid:.4f} ({accuracy_valid*100:.2f}%)\n")
        f.write(f"Macro F1-Score:        {macro_f1_valid:.4f}\n")
        f.write(f"Balanced Accuracy:     {balanced_acc_valid:.4f} ({balanced_acc_valid*100:.2f}%)\n\n")
        
        f.write("-"*70 + "\n")
        f.write("MÉTRICAS POR CLASE - SOLO PREDICCIONES VÁLIDAS\n")
        f.write("-"*70 + "\n\n")
        f.write(df_metrics_valid.to_string())
        f.write("\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CLASES CON MEJOR F1-SCORE - PREDICCIONES VÁLIDAS (Top 10)\n")
        f.write("-"*70 + "\n\n")
        best_f1_valid = df_metrics_valid.nlargest(10, 'F1')
        f.write(best_f1_valid[['F1', 'Precision', 'Recall', 'Support']].to_string())
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("DISTRIBUCIÓN DE CLASES (Support)\n")
        f.write("="*70 + "\n\n")
        
        f.write("Todas las muestras:\n")
        support_stats_all = df_metrics_all['Support'].describe()
        f.write(support_stats_all.to_string())
        f.write("\n\n")
        
        f.write("Predicciones válidas:\n")
        support_stats_valid = df_metrics_valid['Support'].describe()
        f.write(support_stats_valid.to_string())
        f.write("\n")
    
    print(f"   Reporte completo guardado en: {report_path}")
    
    # 10. Imprimir resumen en consola
    print("\n" + "="*70)
    print("RESULTADOS FINALES")
    print("="*70)
    
    print("\n" + "-"*70)
    print("MÉTRICAS SOBRE TODAS LAS MUESTRAS (comparable con arquinew.py):")
    print("-"*70)
    print(f"{'Accuracy:':<30} {accuracy_all:.4f} ({accuracy_all*100:.2f}%)")
    print(f"{'Macro F1-Score:':<30} {macro_f1_all:.4f}")
    print(f"{'Balanced Accuracy:':<30} {balanced_acc_all:.4f} ({balanced_acc_all*100:.2f}%)")
    
    print("\n" + "-"*70)
    print("MÉTRICAS SOBRE PREDICCIONES VÁLIDAS (sin <unknown>):")
    print("-"*70)
    print(f"{'Accuracy:':<30} {accuracy_valid:.4f} ({accuracy_valid*100:.2f}%)")
    print(f"{'Macro F1-Score:':<30} {macro_f1_valid:.4f}")
    print(f"{'Balanced Accuracy:':<30} {balanced_acc_valid:.4f} ({balanced_acc_valid*100:.2f}%)")
    
    print("\n" + "-"*70)
    print("ESTADÍSTICAS:")
    print("-"*70)
    print(f"{'Total de muestras:':<30} {len(y_true)}")
    print(f"{'Clases en label_map:':<30} {len(classes)}")
    print(f"{'Predicciones correctas:':<30} {sum(yt == yp for yt, yp in zip(y_true, y_pred))}")
    print(f"{'Predicciones incorrectas:':<30} {sum(yt != yp for yt, yp in zip(y_true, y_pred) if yp is not None)}")
    print(f"{'Predicciones fuera de label_map:':<30} {sum(yp is None for yp in y_pred)}")
    
    print("\n" + "="*70)
    print(f"Todos los resultados guardados en: {output_path}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis de clasificación multiclase de siteswaps')
    parser.add_argument('--file1', default='siteswap_results.json',
                       help='Primer archivo JSON de resultados')
    parser.add_argument('--file2', default='GIF siteswap_results.json',
                       help='Segundo archivo JSON de resultados')
    parser.add_argument('--label-map', default='../label_map.json',
                       help='Ruta al label_map.json')
    parser.add_argument('--output', default='classification_results',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    main(args.file1, args.file2, args.label_map, args.output)