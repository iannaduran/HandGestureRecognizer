import mediapipe as mp
import json, cv2, os, csv
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "hagrid", "eval_dataset")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "src", "evaluation_results.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "src", "Results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_summary.pdf")

#Loading our Hand Gesture Model
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.IMAGE
)

recognizer = GestureRecognizer.create_from_options(options)

#HaGRID Dataset preparation and loading

#class scope for project
HAGRID_TO_MEDIAPIPE = {
    "like": "Thumb_Up",
    "dislike": "Thumb_Down",
    "peace": "Victory",
    "one": "Pointing_Up",
    "fist": "Closed_Fist",
    "palm": "Open_Palm"
}
TARGET = set(HAGRID_TO_MEDIAPIPE.keys())

def load_flat_eval_dataset(root):
    samples = []

    for fname in os.listdir(root):
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        cls = fname.split("_")[0]
        if cls not in TARGET:
            continue

        samples.append({
            "path": os.path.join(root, fname),
            "label": HAGRID_TO_MEDIAPIPE[cls]
        })

    return samples


#Evaluation Metrics

def evaluate(samples, recognizer):
    y_true, y_pred = [], []

    for s in samples:
        img = cv2.imread(s["path"])
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(
            image_format=mp_image.ImageFormat.SRGB,
            data=img
        )

        result = recognizer.recognize(mp_img)
        pred = result.gestures[0][0].category_name if result.gestures else "None"

        y_true.append(s["label"])
        y_pred.append(pred)

    # Overall
    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")

    return y_true, y_pred

def per_gesture_accuracy(y_true, y_pred):
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    gesture_acc_rows = []
    print("\nPer-Gesture Accuracy:")
    for i, lbl in enumerate(labels):
        correct = cm[i, i]
        total = cm[i].sum()
        acc = (correct / total) * 100 if total > 0 else 0
        print(f"  {lbl}: {acc:.2f}% (n={total})")
        gesture_acc_rows.append({
            "gesture": lbl,
            "accuracy_percent": f"{acc:.2f}",
            "n_samples": total
        })
    return labels, cm, gesture_acc_rows

#Confusion Matrix
def plot_cm(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

#Save results
def save_evaluation_report(labels, cm, gesture_acc_rows, output_path):
    with PdfPages(output_path) as pdf:

        #PAGE 1: Per-Gesture Accuracy
        df = pd.DataFrame(gesture_acc_rows)

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.axis('off')
        table = ax1.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        ax1.set_title("Per-Gesture Accuracy", fontsize=14, pad=20)
        pdf.savefig(fig1)
        plt.close(fig1)

        #PAGE 2: Confusion Matrix
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            ax=ax2
        )
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_title("Confusion Matrix")
        plt.tight_layout()

        pdf.savefig(fig2)
        plt.close(fig2)

    print(f"\nEvaluation report saved to {output_path}")

#Run
samples = load_flat_eval_dataset(DATA_DIR)
print("Loaded samples:", len(samples))
y_true, y_pred = evaluate(samples, recognizer)
labels = [
    "Thumb_Up", "Thumb_Down", "Victory",
    "Pointing_Up", "Closed_Fist", "Open_Palm"
]
labels, cm, gesture_acc_rows = per_gesture_accuracy(y_true, y_pred)
save_evaluation_report(labels, cm, gesture_acc_rows, RESULTS_FILE)