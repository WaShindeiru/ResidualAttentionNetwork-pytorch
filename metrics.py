!pip install scikit-learn
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import tempfile
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def count_flops_per_second(model, input_data):
    try:
        input_shape = input_data.shape
        # 1. Konwersja modelu do stałego grafu
        concrete_func = tf.function(lambda x: model(x))
        concrete_func = concrete_func.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        # 2. Załaduj graph_def do sesji
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            flops = tf.compat.v1.profiler.profile(
                graph=graph,
                run_meta=run_meta,
                cmd='op',
                options=opts
            )

        total_flops = flops.total_float_ops

        # 3. Czas wykonania predykcji
        start = time.time()
        _ = model.predict(input_data)
        end = time.time()
        inference_time = end - start
        # 4. FLOPS per second
        flops_per_second = total_flops / inference_time
        print(f"🔧 FLOPS/s: {flops_per_second:,.2f}")
    except Exception as e:
        print(f"❌ Nie udało się obliczyć FLOPS/s: {e}")





def Show_Model_Statistics(cnn_model, history, class_names=None):
    # Ewaluacja i predykcja
    loss, accuracy = cnn_model.evaluate(x_test_cnn, y_test_cat)
    y_pred = cnn_model.predict(x_test_cnn)
    mse = mean_squared_error(y_test_cat, y_pred)

    print(f'\n📈 Test accuracy: {accuracy:.4f}')
    print(f'📉 Mean Squared Error: {mse:.4f}')

    # Czas i przepustowość
    start_time = time.time()
    _ = cnn_model.predict(x_test_cnn)
    end_time = time.time()
    elapsed = end_time - start_time
    throughput = len(x_test_cnn) / elapsed
    print(f'⏱️ Czas predykcji: {elapsed:.4f} sekund')
    print(f'⚡ Przepustowość: {throughput:.2f} próbek/sekundę')

    # FLOPS
    print("\n⚙️ Szacowanie FLOPS:")
    count_flops_per_second(cnn_model, x_test_cnn)

    # Parametry modelu
    print(f"\n📦 Liczba parametrów w modelu: {cnn_model.count_params():,}")

    # Rozmiar modelu
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
        tmp_file_path = tmp_file.name.replace(".h5", ".keras")
        cnn_model.save(tmp_file_path)
        model_size = os.path.getsize(tmp_file.name) / (1024 ** 2)
    print(f"💾 Rozmiar modelu: {model_size:.2f} MB")

    # Wykresy: Accuracy i Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ROC Curve
    y_pred_prob = y_pred.ravel()
    fpr, tpr, thresholds = roc_curve(y_test_cat.ravel(), y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Macierz błędów i klasyfikacja
    if class_names is None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    y_true = y_test_cat.argmax(axis=1)
    y_pred_classes = y_pred.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Przewidziane klasy")
    plt.ylabel("Rzeczywiste klasy")
    plt.title("Macierz błędów")
    plt.show()

    # Precision, Recall, F1
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    print(f"\n🔍 Precision (weighted): {precision:.4f}")
    print(f"🔍 Recall (weighted): {recall:.4f}")
    print(f"🔍 F1-score (weighted): {f1:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # Overfitting check
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    acc_diff = abs(train_acc - val_acc)
    print(f"\n🔎 Overfitting delta (Train - Val acc): {acc_diff:.4f}")
    if acc_diff > 0.05:
        print("⚠️ Możliwe przeuczenie modelu.")
    else:
        print("✅ Brak istotnego przeuczenia.")

