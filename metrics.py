import os
import tempfile
import torch

from IPython.core.display_functions import display
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_curve, auc, \
    mean_squared_error, confusion_matrix
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def _calculate_flops(model, input_data):

    try:
        concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
            tf.TensorSpec(input_data.shape, model.inputs[0].dtype)
        )
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd='op', options=opts
            )
        return flops.total_float_ops if flops else 0
    except Exception as e:
        print(f"⚠️ Ostrzeżenie: Nie udało się obliczyć FLOPS. Błąd: {e}")
        return 0
def _get_model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp_file:
        model.save(tmp_file.name)
        model_size_mb = os.path.getsize(tmp_file.name) / (1024 ** 2)
    return model_size_mb

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compare_models(models_dict, x_test, y_test_cat, device):
    """
    Porównuje wiele modeli, generując tabelę wyników, wykresy porównawcze,
    krzywe uczenia, krzywe ROC i macierze błędów.
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    results = []
    y_true_labels = y_test_cat
    y_test_cat = np.array(y_test_cat)
    y_true_labels = torch.tensor(y_true_labels, device=device)
    # y_true_labels = np.argmax(y_test_cat, axis=1)

    # KROK 1: ZBIERANIE DANYCH I PREdykcji
    print("--- Rozpoczynanie analizy modeli ---")
    for name, data in models_dict.items():
        print(f"Analizowanie: {name}...")
        model = data['model'].to(device)
        history = data['history']
        transform_ = data['transform']
        x_transformed = [transform_(x_) for x_ in x_test]
        x_test = torch.stack(x_transformed).to(device)

        start_time = time.time()
        y_pred_proba = model(x_test)
        inference_time = time.time() - start_time

        data['y_pred_proba'] = y_pred_proba
        y_pred_classes = torch.argmax(y_pred_proba, dim=1)

        accuracy = y_pred_classes.eq(y_true_labels).float().mean().item()

        # fpr_micro, tpr_micro, _ = roc_curve(y_test_cat.ravel(), y_pred_proba.ravel())
        # roc_auc_micro = auc(fpr_micro, tpr_micro)

        throughput = len(x_test) / inference_time
        precision = precision_score(y_true_labels.cpu().numpy(), y_pred_classes.cpu().numpy(), average='weighted')
        recall = recall_score(y_true_labels.cpu().numpy(), y_pred_classes.cpu().numpy(), average='weighted')
        f1 = f1_score(y_true_labels.cpu().numpy(), y_pred_classes.cpu().numpy(), average='weighted')
        a = y_test_cat
        b = y_pred_classes.detach().cpu().numpy()
        print(a.shape)
        print(b.shape)
        mse = mean_squared_error(a, b)
        params, _ = count_parameters(model)
        # model_size_mb = _get_model_size_mb(model)
        total_flops = _calculate_flops(model, x_test[:1])
        train_acc = history.get('train_acc', [0])[-1]
        val_acc = history.get('val_accuracy', [0])[-1]
        overfit_delta = abs(train_acc - val_acc)

        results.append({
            'Model': name, 'Accuracy': accuracy, 'F1-score (w)': f1,
            # 'Model': name, 'Accuracy': accuracy, 'AUC (micro)': roc_auc_micro, 'F1-score (w)': f1,
            'Precision (w)': precision, 'Recall (w)': recall, 'Czas predykcji (s)': inference_time,
            'Przepustowość (próbki/s)': throughput, 'Liczba parametrów': params,
            'FLOPS': total_flops,
            # 'Rozmiar (MB)': model_size_mb, 'FLOPS': total_flops,
            'Przeuczenie (delta)': overfit_delta, 'MSE': mse,
        })

    results_df = pd.DataFrame(results).set_index('Model')

    # KROK 2: TABELA I WYKRESY SŁUPKOWE
    higher_is_better = ['Accuracy', 'AUC (micro)', 'F1-score (w)', 'Precision (w)', 'Recall (w)', 'Przepustowość (próbki/s)']
    lower_is_better = ['Czas predykcji (s)', 'Liczba parametrów', 'Rozmiar (MB)', 'FLOPS', 'Przeuczenie (delta)', 'MSE']

    styled_df = results_df.style.background_gradient(cmap='Greens', subset=higher_is_better) \
                               .background_gradient(cmap='Greens_r', subset=lower_is_better) \
                               .format('{:.4f}', subset=pd.IndexSlice[:, ['Accuracy', 'AUC (micro)', 'F1-score (w)', 'Precision (w)', 'Recall (w)', 'Przeuczenie (delta)', 'MSE']]) \
                               .format({'Czas predykcji (s)': '{:.3f}', 'Przepustowość (próbki/s)': '{:,.0f}', 'Liczba parametrów': '{:,}', 'Rozmiar (MB)': '{:.2f}', 'FLOPS': '{:,}'})

    print("\n\n" + "="*50)
    print("🏆 TABELA PORÓWNAWCZA MODELI 🏆")
    print("="*50)
    display(styled_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Wizualne Porównanie Modeli', fontsize=16)
    sns.barplot(x=results_df.index, y=results_df['Accuracy'], ax=axes[0, 0], palette='viridis', hue=results_df.index, legend=False)
    axes[0, 0].set_title('Dokładność (Accuracy)')
    sns.barplot(x=results_df.index, y=results_df['F1-score (w)'], ax=axes[0, 1], palette='plasma', hue=results_df.index, legend=False)
    axes[0, 1].set_title('F1-score (ważony)')
    sns.barplot(x=results_df.index, y=results_df['Przepustowość (próbki/s)'], ax=axes[1, 0], palette='magma', hue=results_df.index, legend=False)
    axes[1, 0].set_title('Wydajność (Przepustowość)')
    sns.barplot(x=results_df.index, y=results_df['Rozmiar (MB)'], ax=axes[1, 1], palette='cividis', hue=results_df.index, legend=False)
    axes[1, 1].set_title('Zasoby (Rozmiar modelu)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # KROK 3: PORÓWNANIE KRZYWYCH UCZENIA (Wersja z ujednoliconymi kolorami)
    print("\n\n" + "="*50)
    print("🧠 PORÓWNANIE KRZYWYCH UCZENIA 🧠")
    print("="*50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    num_models = len(models_dict)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, num_models)]

    for (name, data), color in zip(models_dict.items(), colors):
        history = data['history']


        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            ax1.plot(history.history['accuracy'], color=color, linestyle='-', label=f'{name} Train')
            ax1.plot(history.history['val_accuracy'], color=color, linestyle='--', label=f'{name} Val')

        if 'loss' in history.history and 'val_loss' in history.history:
            ax2.plot(history.history['loss'], color=color, linestyle='-', label=f'{name} Train')
            ax2.plot(history.history['val_loss'], color=color, linestyle='--', label=f'{name} Val')

    ax1.set_title('Porównanie Dokładności (Accuracy)')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Dokładność')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_title('Porównanie Straty (Loss)')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Strata')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # KROK 4: PORÓWNANIE KRZYWYCH ROC (MICRO-AVERAGE)
    print("\n\n" + "="*50)
    print("📈 PORÓWNANIE KRZYWYCH ROC (Uśrednione) 📈")
    print("="*50)
    plt.figure(figsize=(10, 8))
    for name, data in models_dict.items():
        y_pred_proba = data['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test_cat.ravel(), y_pred_proba.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Losowy klasyfikator')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Porównanie uśrednionych krzywych ROC (Micro-Average)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # KROK 5: PORÓWNANIE MACIERZY BŁĘDÓW
    print("\n\n" + "="*50)
    print("🔢 PORÓWNANIE MACIERZY BŁĘDÓW 🔢")
    print("="*50)
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, models_dict.items()):
        y_pred_classes = np.argmax(data['y_pred_proba'], axis=1)
        cm = confusion_matrix(y_true_labels, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Macierz błędów dla: {name}')
        ax.set_xlabel('Przewidziane klasy')
        ax.set_ylabel('Rzeczywiste klasy')
    plt.tight_layout()
    plt.show()

def Show_Model_Statistics(cnn_model, history, x_test_cnn, y_test_cat, class_names=None):
    """
    Wyświetla szczegółowe statystyki i wizualizacje dla pojedynczego modelu.
    """
    loss, accuracy = cnn_model.evaluate(x_test_cnn, y_test_cat, verbose=0)
    y_pred = cnn_model.predict(x_test_cnn)
    mse = mean_squared_error(y_test_cat, y_pred)

    fpr, tpr, _ = roc_curve(y_test_cat.ravel(), y_pred.ravel())
    roc_auc_micro = auc(fpr, tpr)

    print(f'\n📈 Test accuracy: {accuracy:.4f}')
    print(f'📊 Pole pod krzywą ROC (AUC micro): {roc_auc_micro:.4f}')
    print(f'📉 Mean Squared Error: {mse:.4f}')

    start_time = time.time()
    _ = cnn_model.predict(x_test_cnn, verbose=0)
    end_time = time.time()
    elapsed = end_time - start_time
    throughput = len(x_test_cnn) / elapsed
    print(f'⏱️ Czas predykcji: {elapsed:.4f} sekund')
    print(f'⚡ Przepustowość: {throughput:.2f} próbek/sekundę')

    print("\n⚙️ Szacowanie FLOPS:")
    if len(x_test_cnn) > 0:
        input_data_flops = x_test_cnn[:1]
    else:
        input_shape = cnn_model.inputs[0].shape[1:]
        input_data_flops = tf.random.uniform([1] + list(input_shape))

    total_flops = _calculate_flops(cnn_model, input_data_flops)
    print(f"Total estimated FLOPS: {total_flops:,}")

    print(f"\n📦 Liczba parametrów w modelu: {cnn_model.count_params():,}")

    model_size_mb = _get_model_size_mb(cnn_model)
    print(f"💾 Rozmiar modelu: {model_size_mb:.2f} MB")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        print("⚠️ Warning: 'accuracy' or 'val_accuracy' not found in history.")


    plt.subplot(1, 2, 2)
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    else:
         print("⚠️ Warning: 'loss' or 'val_loss' not found in history.")

    plt.tight_layout()
    plt.show()
    print("\n📊 ROC Curve (per class):")
    plt.figure(figsize=(10, 8))
    if class_names is None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    for i in range(y_test_cat.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve for class {class_names[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Per Class)')
    plt.legend(loc="lower right")
    plt.show()

    if class_names is None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    y_true = y_test_cat.argmax(axis=1)
    y_pred_classes = y_pred.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Przewidziane klasy")
    plt.ylabel("Rzeczywiste klasy")
    plt.title("Macierz błędów")
    plt.tight_layout()
    plt.show()

    try:
        precision = precision_score(y_true, y_pred_classes, average='weighted')
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        print(f"\n🔍 Precision (weighted): {precision:.4f}")
        print(f"🔍 Recall (weighted): {recall:.4f}")
        print(f"🔍 F1-score (weighted): {f1:.4f}")

        print("\n📊 Classification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=class_names))

    except Exception as e:
        print(f"⚠️ Warning: Could not calculate classification metrics. Error: {e}")


    train_acc = history.history.get('accuracy', [0])[-1]
    val_acc = history.history.get('val_accuracy', [0])[-1]
    acc_diff = abs(train_acc - val_acc)
    print(f"\n🔎 Overfitting delta (Train - Val acc): {acc_diff:.4f}")
    if acc_diff > 0.05:
        print("⚠️ Możliwe przeuczenie modelu.")
    else:
        print("✅ Brak istotnego przeuczenia.")
