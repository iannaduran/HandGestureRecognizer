import os, cv2, numpy as np, mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from mediapipe.tasks.python.vision import (
    GestureRecognizer, GestureRecognizerOptions,
    HandLandmarker, HandLandmarkerOptions
)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(ROOT, "hagrid", "eval_dataset")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
GESTURE_MODEL = os.path.join(PROJECT_ROOT, "src", "gesture_recognizer.task")
HAND_MODEL = os.path.join(ROOT, "src", "hand_landmarker.task")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "src", "Results")
LANDMARK_PDF = os.path.join(
    RESULTS_DIR,
    "landmark_normalization_ablation_report.pdf"
)

#class scope for project
HAGRID_TO_MEDIAPIPE = {
    "like": "Thumb_Up",
    "dislike": "Thumb_Down",
    "peace": "Victory",
    "point": "Pointing_Up",
    "fist": "Closed_Fist",
    "palm": "Open_Palm"
}

#Load Tasks models
hand_landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )
)

recognizer = GestureRecognizer.create_from_options(
    GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
        running_mode=VisionRunningMode.IMAGE
    )
)


#Normalization

def normalize_centered(lm):
    return lm - np.mean(lm, axis=0)

def normalize_scale(lm):
    lm = normalize_centered(lm)
    d = np.max(np.linalg.norm(lm, axis=1))
    return lm if d < 1e-6 else lm / d

def normalize_rotation(lm):
    wrist, middle = lm[0], lm[9]
    v = middle - wrist
    if np.linalg.norm(v[:2]) < 1e-6:
        return lm - wrist
    ang = -np.arctan2(v[1], v[0])
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang),  np.cos(ang)]])
    xy = (lm[:, :2] - wrist[:2]) @ R.T
    return np.hstack([xy, lm[:, 2:3]])


#Render landmarks to image

def render_landmarks(lm, size=224):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    xy = lm[:, :2]
    xy -= xy.min(axis=0)
    xy /= (xy.max(axis=0) + 1e-6)
    xy = xy * size * 0.8 + size * 0.1
    for x, y in xy.astype(int):
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
    return img


#Ablation Study
strategies = {
    "Raw": lambda x: x,
    "Centered": normalize_centered,
    "Scaled": normalize_scale,
    "Rotated": normalize_rotation,
    "Centered+Scaled": lambda x: normalize_scale(normalize_centered(x)),
    "All": lambda x: normalize_rotation(normalize_scale(normalize_centered(x)))
}

results = {}

for name, norm_fn in strategies.items():
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

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_res = hand_landmarker.detect(mp_img)
        if not hand_res.hand_landmarks:
            continue

        lm = np.array([[p.x, p.y, p.z] for p in hand_res.hand_landmarks[0]])
        lm = norm_fn(lm)
        synth = render_landmarks(lm)

        synth_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=synth)
        out = recognizer.recognize(synth_mp)

        if out.gestures:
            y_true.append(label)
            y_pred.append(out.gestures[0][0].category_name)

    acc = accuracy_score(y_true, y_pred)
    results[name] = acc
    print(f"{name:25s}: {acc*100:.2f}%")


#Save to PDF
with PdfPages(LANDMARK_PDF) as pdf:

    #PAGE 1: Accuracy Table
    df = pd.DataFrame({
        "Strategy": list(results.keys()),
        "Accuracy (%)": [v * 100 for v in results.values()]
    })

    fig1, ax1 = plt.subplots(figsize=(8, 4))
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

    ax1.set_title(
        "Landmark Normalization Ablation Results",
        fontsize=14,
        pad=25
    )

    plt.subplots_adjust(top=0.85)

    pdf.savefig(fig1)
    plt.close(fig1)

    #PAGE 2: Bar Chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(results.keys(), [v * 100 for v in results.values()])
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Normalization Ablation on HaGRID (Vision Pipeline)")
    ax2.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    pdf.savefig(fig2)
    plt.close(fig2)

print(f"\nLandmark ablation report saved to {LANDMARK_PDF}")