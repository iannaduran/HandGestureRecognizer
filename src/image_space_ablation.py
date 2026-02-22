import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import mediapipe as mp
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(ROOT, "hagrid", "eval_dataset")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "gesture_recognizer.task")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "src", "Results")
ABLATION_PDF = os.path.join(RESULTS_DIR, "image_space_ablation_report.pdf")


#class scope for project
HAGRID_TO_MEDIAPIPE = {
    "like": "Thumb_Up",
    "dislike": "Thumb_Down",
    "peace": "Victory",
    "point": "Pointing_Up",
    "fist": "Closed_Fist",
    "palm": "Open_Palm"
}


#Load Model
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions

recognizer = GestureRecognizer.create_from_options(
    GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE
    )
)


# Image-space Normalization

def transform_centered(img):
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    size = int(min(w, h) * 0.8)
    x1 = max(cx - size//2, 0)
    y1 = max(cy - size//2, 0)
    x2 = min(cx + size//2, w)
    y2 = min(cy + size//2, h)
    return cv2.resize(img[y1:y2, x1:x2], (w, h))

def transform_scaled(img, scale=1.2):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, 0, scale)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def transform_rotated(img, angle=15):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def transform_combined(img):
    img = transform_centered(img)
    img = transform_scaled(img, scale=1.1)
    img = transform_rotated(img, angle=10)
    return img


# Ablation Study
strategies = {
    "Raw": lambda x: x,
    "Centered": transform_centered,
    "Scaled": transform_scaled,
    "Rotated": transform_rotated,
    "Combined": transform_combined
}

results = {}

for name, fn in strategies.items():
    y_true, y_pred = [], []
    for file in os.listdir(IMG_PATH):
        label = None
        for k in HAGRID_TO_MEDIAPIPE:
            if file.startswith(k):
                label = HAGRID_TO_MEDIAPIPE[k]
                break
        if label is None:
            continue

        img = cv2.imread(os.path.join(IMG_PATH, file))
        if img is None:
            continue

        img_t = fn(img)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_t)

        out = recognizer.recognize(mp_img)
        if out.gestures:
            y_true.append(label)
            y_pred.append(out.gestures[0][0].category_name)

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    results[name] = acc
    print(f"{name:15s}: {acc*100:.2f}%")


# Save to PDF
with PdfPages(ABLATION_PDF) as pdf:

    #PAGE 1: Accuracy Table
    df = pd.DataFrame({
        "Strategy": list(results.keys()),
        "Accuracy (%)": [v * 100 for v in results.values()]
    })

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.axis("off")

    table = ax1.table(
        cellText=df.round(2).values,
        colLabels=df.columns,
        loc="center",
        bbox=[0, 0.1, 1, 0.7]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax1.set_title("Image-Space Normalization Ablation Results",
                  fontsize=14, pad=25)

    plt.subplots_adjust(top=0.85)

    pdf.savefig(fig1)
    plt.close(fig1)

    #PAGE 2: Bar Chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(results.keys(), [v * 100 for v in results.values()])
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Image-Space Normalization Ablation on HaGRID Subset")
    ax2.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    pdf.savefig(fig2)
    plt.close(fig2)

print(f"\nAblation report saved to {ABLATION_PDF}")
