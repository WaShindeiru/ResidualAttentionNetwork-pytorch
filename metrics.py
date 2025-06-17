# POCZĄTEK PLIKU METRICS.PY
import os
import time
import tempfile
import warnings
import io  # <-- KLUCZOWY IMPORT

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.core.display_functions import display
from sklearn.metrics import (classification_report, precision_score, recall_score,
                             f1_score, roc_curve, auc, mean_squared_error, confusion_matrix)

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("Ostrzeżenie: Biblioteka fvcore nie jest zainstalowana. Nie będzie można obliczyć FLOPS.")
    print("Aby ją zainstalować, uruchom: pip install fvcore")
    FlopCountAnalysis = None


def _calculate_pytorch_flops(model, input_tensor):
    """Oblicza FLOPS dla modelu PyTorch."""
    if FlopCountAnalysis is None:
        return 0
    try:
        model.eval()
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total()
    except Exception as e:
        warnings.warn(f"Nie udało się obliczyć FLOPS. Błąd: {e}")
        return 0


def _get_pytorch_model_size_mb(model):
    """Oblicza rozmiar modelu PyTorch w MB, zapisując go do bufora w pamięci."""
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_size_mb = buffer.tell() / (1024 ** 2)
        return model_size_mb
    except Exception as e:
        warnings.warn(f"Nie udało się obliczyć rozmiaru modelu. Błąd: {e}")
        return 0


def _count_pytorch_parameters(model):
    """Zlicza parametry w modelu PyTorch."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compare_models(models_dict, x_test_raw, y_test_labels, device):
    """
    Porównuje wiele modeli PyTorch, generując tabelę wyników, wykresy porównawcze,
    krzywe uczenia, krzywe ROC i macierze błędów.
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    results = []

    y_true_tensor = torch.tensor(y_test_labels, dtype=torch.long).to(device)
    y_true_one_hot_np = np.eye(len(class_names))[y_test_labels]

    print("--- Rozpoczynanie analizy modeli ---")
    for name, data in models_dict.items():
        print(f"Analizowanie: {name}...")
        model = data['model'].to(device)
        history = data.get('history', {})
        transform = data['transform']

        model.eval()

        x_transformed = [transform(x) for x in x_test_raw]
        x_test_tensor = torch.stack(x_transformed).to(device)

        with torch.no_grad():
            start_time = time.time()

            if name != "VGG":
                y_pred_logits = model(x_test_tensor)
            else:
                y_pred_logits, _, _ = model(x_test_tensor)

            inference_time = time.time() - start_time

            y_pred_proba = torch.softmax(y_pred_logits, dim=1)
            y_pred_classes = torch.argmax(y_pred_logits, dim=1)

        y_true_np = y_true_tensor.cpu().numpy()
        y_pred_classes_np = y_pred_classes.cpu().numpy()
        y_pred_proba_np = y_pred_proba.cpu().numpy()

        accuracy = (y_pred_classes == y_true_tensor).float().mean().item()

        fpr_micro, tpr_micro, _ = roc_curve(y_true_one_hot_np.ravel(), y_pred_proba_np.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        throughput = len(x_test_tensor) / inference_time
        precision = precision_score(y_true_np, y_pred_classes_np, average='weighted', zero_division=0)
        recall = recall_score(y_true_np, y_pred_classes_np, average='weighted', zero_division=0)
        f1 = f1_score(y_true_np, y_pred_classes_np, average='weighted', zero_division=0)
        mse = mean_squared_error(y_true_np, y_pred_classes_np)

        params, _ = _count_pytorch_parameters(model)
        model_size_mb = _get_pytorch_model_size_mb(model)  # Ta linia teraz użyje poprawionej funkcji
        total_flops = _calculate_pytorch_flops(model, x_test_tensor[:1])

        train_acc = history.get('train_acc', [0])[-1]
        val_acc = history.get('val_acc', [0])[-1]
        overfit_delta = abs(train_acc - val_acc)

        data['y_pred_proba_np'] = y_pred_proba_np
        data['y_pred_classes_np'] = y_pred_classes_np

        results.append({
            'Model': name, 'Accuracy': accuracy, 'AUC (micro)': roc_auc_micro, 'F1-score (w)': f1,
            'Precision (w)': precision, 'Recall (w)': recall, 'Czas predykcji (s)': inference_time,
            'Przepustowość (próbki/s)': throughput, 'Liczba parametrów': params,
            'Rozmiar (MB)': model_size_mb, 'FLOPS': total_flops,
            'Przeuczenie (delta)': overfit_delta, 'MSE': mse,
        })

    results_df = pd.DataFrame(results).set_index('Model')

    higher_is_better = ['Accuracy', 'AUC (micro)', 'F1-score (w)', 'Precision (w)', 'Recall (w)',
                        'Przepustowość (próbki/s)']
    lower_is_better = ['Czas predykcji (s)', 'Liczba parametrów', 'Rozmiar (MB)', 'FLOPS', 'Przeuczenie (delta)', 'MSE']

    valid_higher = [col for col in higher_is_better if col in results_df.columns]
    valid_lower = [col for col in lower_is_better if col in results_df.columns]

    styled_df = results_df.style.background_gradient(cmap='Greens', subset=valid_higher) \
        .background_gradient(cmap='Greens_r', subset=valid_lower) \
        .format('{:.4f}', subset=[c for c in ['Accuracy', 'AUC (micro)', 'F1-score (w)', 'Precision (w)', 'Recall (w)',
                                              'Przeuczenie (delta)', 'MSE'] if c in results_df.columns]) \
        .format({'Czas predykcji (s)': '{:.3f}', 'Przepustowość (próbki/s)': '{:,.0f}', 'Liczba parametrów': '{:,}',
                 'Rozmiar (MB)': '{:.2f}', 'FLOPS': '{:,.0f}'})

    print("\n\n" + "=" * 50)
    print("TABELA PORÓWNAWCZA MODELI")
    print("=" * 50)
    display(styled_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Wizualne Porównanie Modeli', fontsize=16)
    sns.barplot(x=results_df.index, y=results_df['Accuracy'], ax=axes[0, 0], palette='viridis', hue=results_df.index,
                legend=False)
    axes[0, 0].set_title('Dokładność (Accuracy)')
    sns.barplot(x=results_df.index, y=results_df['F1-score (w)'], ax=axes[0, 1], palette='plasma', hue=results_df.index,
                legend=False)
    axes[0, 1].set_title('F1-score (ważony)')
    sns.barplot(x=results_df.index, y=results_df['Przepustowość (próbki/s)'], ax=axes[1, 0], palette='magma',
                hue=results_df.index, legend=False)
    axes[1, 0].set_title('Wydajność (Przepustowość)')
    sns.barplot(x=results_df.index, y=results_df['Rozmiar (MB)'], ax=axes[1, 1], palette='cividis',
                hue=results_df.index, legend=False)
    axes[1, 1].set_title('Zasoby (Rozmiar modelu)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n\n" + "=" * 50)
    print("PORÓWNANIE KRZYWYCH UCZENIA")
    print("=" * 50)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    num_models = len(models_dict)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, num_models)]

    for (name, data), color in zip(models_dict.items(), colors):
        history = data.get('history', {})
        if history:
            # ax1.plot(history.get('train_acc', []), color=color, linestyle='-', label=f'{name} Train Acc')
            ax1.plot(history.get('val_acc', []), color=color, linestyle='--', label=f'{name} Val Acc')
            # ax2.plot(history.get('loss', []), color=color, linestyle='-', label=f'{name} Train Loss')
            ax2.plot(history.get('val_loss', []), color=color, linestyle='--', label=f'{name} Val Loss')

    ax1.set_title('Porównanie Dokładności (Accuracy)');
    ax1.set_xlabel('Epoka');
    ax1.set_ylabel('Dokładność')
    ax1.legend();
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Porównanie Straty (Loss)');
    ax2.set_xlabel('Epoka');
    ax2.set_ylabel('Strata')
    ax2.legend();
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout();
    plt.show()

    print("\n\n" + "=" * 50)
    print("PORÓWNANIE KRZYWYCH ROC (Uśrednione)")
    print("=" * 50)
    plt.figure(figsize=(10, 8))
    for name, data in models_dict.items():
        y_pred_proba_np = data['y_pred_proba_np']
        fpr, tpr, _ = roc_curve(y_true_one_hot_np.ravel(), y_pred_proba_np.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Losowy klasyfikator')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('Porównanie uśrednionych krzywych ROC (Micro-Average)')
    plt.legend(loc="lower right");
    plt.grid(True);
    plt.show()

    print("\n\n" + "=" * 50)
    print("PORÓWNANIE MACIERZY BŁĘDÓW")
    print("=" * 50)
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), squeeze=False)
    axes = axes.flatten()

    for ax, (name, data) in zip(axes, models_dict.items()):
        y_pred_classes_np = data['y_pred_classes_np']
        cm = confusion_matrix(y_true_np, y_pred_classes_np)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Macierz błędów dla: {name}')
        ax.set_xlabel('Przewidziane klasy');
        ax.set_ylabel('Rzeczywiste klasy')
    plt.tight_layout();
    plt.show()


def show_pytorch_model_stats(model, history, test_loader, device, class_names=None):
    """
    Wyświetla szczegółowe statystyki i wizualizacje dla pojedynczego modelu PyTorch.
    """
    if class_names is None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model.to(device)
    model.eval()

    all_preds, all_labels, all_probas = [], [], []

    start_time = time.time()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            probas = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())

    inference_time = time.time() - start_time

    y_true = np.array(all_labels)
    y_pred_classes = np.array(all_preds)
    y_pred_proba = np.array(all_probas)

    accuracy = np.mean(y_true == y_pred_classes)
    y_true_one_hot = np.eye(len(class_names))[y_true]
    fpr_micro, tpr_micro, _ = roc_curve(y_true_one_hot.ravel(), y_pred_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    mse = mean_squared_error(y_true, y_pred_classes)
    throughput = len(y_true) / inference_time

    print(f'\nTest accuracy: {accuracy:.4f}')
    print(f'Pole pod krzywą ROC (AUC micro): {roc_auc_micro:.4f}')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Czas predykcji: {inference_time:.4f} sekund')
    print(f'Przepustowość: {throughput:.2f} próbek/sekundę')

    sample_input, _ = next(iter(test_loader))
    total_flops = _calculate_pytorch_flops(model, sample_input[:1].to(device))
    print(f"\nSzacowane FLOPS: {total_flops:,.0f}")

    params, _ = _count_pytorch_parameters(model)
    print(f"Liczba parametrów w modelu: {params:,}")

    model_size_mb = _get_pytorch_model_size_mb(model)  # Ta linia teraz użyje poprawionej funkcji
    print(f"Rozmiar modelu: {model_size_mb:.2f} MB")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if history and 'train_acc' in history and 'val_acc' in history:
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.title('Accuracy over Epochs');
        plt.xlabel('Epoch');
        plt.ylabel('Accuracy');
        plt.legend()
    else:
        print("Ostrzeżenie: Brak kluczy 'train_acc' lub 'val_acc' w historii.")

    plt.subplot(1, 2, 2)
    if history and 'train_loss' in history and 'val_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss over Epochs');
        plt.xlabel('Epoch');
        plt.ylabel('Loss');
        plt.legend()
    else:
        print("Ostrzeżenie: Brak kluczy 'train_loss' lub 'val_loss' w historii.")
    plt.tight_layout();
    plt.show()

    print("\nKrzywe ROC (dla każdej klasy):")
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Klasa {class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC (dla każdej z klas)');
    plt.legend(loc="lower right");
    plt.show()

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Przewidziane klasy");
    plt.ylabel("Rzeczywiste klasy");
    plt.title("Macierz błędów")
    plt.tight_layout();
    plt.show()

    print("\nRaport klasyfikacji:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=0))

    if history:
        train_acc = history.get('train_acc', [0])[-1]
        val_acc = history.get('val_acc', [0])[-1]
        acc_diff = abs(train_acc - val_acc)
        print(f"\n🔎 Delta przeuczenia (różnica Train Acc - Val Acc): {acc_diff:.4f}")
        if acc_diff > 0.05:
            print("Możliwe przeuczenie modelu (różnica > 5%).")
        else:
            print("Brak istotnych oznak przeuczenia.")

