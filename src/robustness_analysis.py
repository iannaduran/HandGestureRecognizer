import mediapipe as mp
import cv2, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
ROBUSTNESS_PDF = os.path.join(RESULTS_DIR, "robustness_report.pdf")
MODEL_PATH = os.path.join(BASE_DIR, "gesture_recognizer.task")

# ── Model setup ───────────────────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)
recognizer = GestureRecognizer.create_from_options(options)

LABELS = ["Thumb_Up", "Thumb_Down", "Victory", "Pointing_Up", "Closed_Fist", "Open_Palm"]

# ── Dataset loading ───────────────────────────────────────────────────────────
def load_dataset(root):
    samples = []
    for fname in os.listdir(root):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        stem = fname.replace("screenshot_", "").rsplit(".", 1)[0]
        parts = stem.split("_")
        label_parts = []
        for part in parts:
            if re.match(r"^\d{8}-", part):
                break
            label_parts.append(part)
        label = "_".join(label_parts)
        if label not in LABELS:
            continue
        samples.append({"path": os.path.join(root, fname), "label": label})
    return samples


# ── Transforms ────────────────────────────────────────────────────────────────
def apply_motion_blur(img, kernel_size):
    """Horizontal smear — simulates hand movement during capture."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_occlusion(img, fraction):
    """
    Black out a rectangle covering `fraction` of the image area, placed at
    the centre — simulates an object partially hiding the hand.
    Position is fixed (not random) so results are reproducible.
    """
    h, w = img.shape[:2]
    block_side = int(min(w, h) * fraction)
    x = (w - block_side) // 2
    y = (h - block_side) // 2
    out = img.copy()
    out[y:y + block_side, x:x + block_side] = 0
    return out


def apply_gaussian_noise(img, std):
    """
    Add zero-mean Gaussian noise — proxy for background clutter and
    low-quality camera sensors.
    """
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_with_transform(samples, recognizer, transform_fn=None):
    y_true, y_pred = [], []
    for s in samples:
        img = cv2.imread(s["path"])
        if img is None:
            continue
        if transform_fn is not None:
            img = transform_fn(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=img_rgb)
        result = recognizer.recognize(mp_img)
        pred = result.gestures[0][0].category_name if result.gestures else "None"
        y_true.append(s["label"])
        y_pred.append(pred)
    return accuracy_score(y_true, y_pred) * 100


# ── Visualisation helpers ─────────────────────────────────────────────────────
def make_preview_page(samples):
    """
    One row of images: Original | Blur (k=21) | Occlusion (40%) | Noise (std=50)
    Gives the reader a visual sense of each failure condition.
    """
    if not samples:
        return None

    demo_bgr = cv2.imread(samples[0]["path"])
    demo_rgb = cv2.cvtColor(demo_bgr, cv2.COLOR_BGR2RGB)

    examples = [
        ("Original",          demo_rgb),
        ("Motion Blur\n(kernel=21)",   cv2.cvtColor(apply_motion_blur(demo_bgr, 21),       cv2.COLOR_BGR2RGB)),
        ("Occlusion\n(40%)",           cv2.cvtColor(apply_occlusion(demo_bgr, 0.40),        cv2.COLOR_BGR2RGB)),
        ("Gaussian Noise\n(std=50)",   cv2.cvtColor(apply_gaussian_noise(demo_bgr, 50),     cv2.COLOR_BGR2RGB)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, img) in zip(axes, examples):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("Robustness Analysis — Perturbation Previews", fontsize=13)
    plt.tight_layout()
    return fig


def make_charts_page(results, baseline_acc):
    """
    Three side-by-side subplots, one per test type.
    Each shows accuracy vs. severity with the baseline as a dashed reference.
    """
    tests = [
        ("Motion Blur",     "Severity",       "#6366F1"),
        ("Occlusion",       "Severity",       "#F59E0B"),
        ("Gaussian Noise",  "Severity",       "#EF4444"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (condition, xlabel, color) in zip(axes, tests):
        rows = [r for r in results if r["Condition"] == condition]
        severities = [r["Severity"] for r in rows]
        accuracies = [float(r["Accuracy (%)"]) for r in rows]

        ax.plot(severities, accuracies, marker="o", color=color, linewidth=2)
        ax.axhline(baseline_acc, linestyle="--", color="#22C55E", linewidth=1.5,
                   label=f"Baseline ({baseline_acc:.1f}%)")
        ax.set_title(condition, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Accuracy vs. Severity — All Robustness Tests", fontsize=13)
    plt.tight_layout()
    return fig


# ── Report ────────────────────────────────────────────────────────────────────
def save_robustness_report(results, baseline_acc, samples, output_path):
    with PdfPages(output_path) as pdf:

        # PAGE 1: Perturbation visual preview
        preview = make_preview_page(samples)
        if preview:
            pdf.savefig(preview)
            plt.close(preview)

        # PAGE 2: Full summary table
        df = pd.DataFrame(results)
        fig_t, ax_t = plt.subplots(figsize=(9, len(results) * 0.45 + 1.5))
        ax_t.axis("off")
        table = ax_t.table(
            cellText=df.values,
            colLabels=df.columns,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        ax_t.set_title("Robustness Analysis — Full Results", fontsize=13, pad=16)
        pdf.savefig(fig_t)
        plt.close(fig_t)

        # PAGE 3: Accuracy vs. severity charts
        charts = make_charts_page(results, baseline_acc)
        pdf.savefig(charts)
        plt.close(charts)

    print(f"\nRobustness report saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)  # keep noise results reproducible across runs

    samples = load_dataset(DATA_DIR)
    print(f"Loaded {len(samples)} samples from {DATA_DIR}")
    if not samples:
        print("No images found — check that src/Dataset contains screenshot_*.png files.")
        return

    results = []

    # Baseline
    baseline_acc = evaluate_with_transform(samples, recognizer)
    print(f"\nBaseline accuracy: {baseline_acc:.2f}%")

    # ── Motion blur ───────────────────────────────────────────────────────────
    print("\nRunning motion blur test...")
    for ks in [5, 11, 21, 31]:
        acc = evaluate_with_transform(samples, recognizer,
                                      lambda img, k=ks: apply_motion_blur(img, k))
        drop = baseline_acc - acc
        print(f"  kernel={ks:>2}px  →  {acc:.2f}%  (drop: {drop:.2f}%)")
        results.append({"Condition": "Motion Blur",    "Severity": ks,
                         "Accuracy (%)": f"{acc:.2f}", "Drop (%)": f"{drop:.2f}"})

    # ── Occlusion ─────────────────────────────────────────────────────────────
    print("\nRunning occlusion test...")
    for frac in [0.10, 0.25, 0.40, 0.55]:
        acc = evaluate_with_transform(samples, recognizer,
                                      lambda img, f=frac: apply_occlusion(img, f))
        drop = baseline_acc - acc
        print(f"  fraction={int(frac*100):>2}%  →  {acc:.2f}%  (drop: {drop:.2f}%)")
        results.append({"Condition": "Occlusion",      "Severity": int(frac * 100),
                         "Accuracy (%)": f"{acc:.2f}", "Drop (%)": f"{drop:.2f}"})

    # ── Gaussian noise ────────────────────────────────────────────────────────
    print("\nRunning Gaussian noise test...")
    for std in [15, 30, 50, 75]:
        acc = evaluate_with_transform(samples, recognizer,
                                      lambda img, s=std: apply_gaussian_noise(img, s))
        drop = baseline_acc - acc
        print(f"  std={std:>2}  →  {acc:.2f}%  (drop: {drop:.2f}%)")
        results.append({"Condition": "Gaussian Noise", "Severity": std,
                         "Accuracy (%)": f"{acc:.2f}", "Drop (%)": f"{drop:.2f}"})

    save_robustness_report(results, baseline_acc, samples, ROBUSTNESS_PDF)


if __name__ == "__main__":
    main()
