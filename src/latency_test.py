import time
import cv2
from gesture_recognition import recognizer
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "src", "Results")
LATENCY_PDF = os.path.join(RESULTS_DIR, "latency_report.pdf")

# Latency Evaluation Function
# Measures latency per frame for a live camera feed and returns the list of times in ms.

def measure_live_latency(recognizer, frames=200):
    cap = cv2.VideoCapture(0)
    times = []

    print("Warming up camera...")
    for _ in range(30):
        cap.read()

    print("Measuring latency...")
    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(
            image_format=mp_image.ImageFormat.SRGB,
            data=img
        )

        recognizer.recognize(mp_img)

        times.append((time.time() - start) * 1000)  # milliseconds

    cap.release()
    return times

def save_latency_report(times, avg, min_time, max_time, output_path):
    with PdfPages(output_path) as pdf:

        #PAGE 1: Summary Table
        summary_data = {
            "Metric": ["Average (ms)", "Minimum (ms)", "Maximum (ms)", "Target Met (<=100ms)"],
            "Value": [
                f"{avg:.2f}",
                f"{min_time:.2f}",
                f"{max_time:.2f}",
                "YES" if avg <= 100 else "NO"
            ]
        }

        df_summary = pd.DataFrame(summary_data)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.axis("off")
        table = ax1.table(
            cellText=df_summary.values,
            colLabels=df_summary.columns,
            loc="center",
            bbox=[0, 0.05, 1, 0.7]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        ax1.set_title("Latency Evaluation Summary", fontsize=14, pad=20)
        plt.subplots_adjust(top=0.85)
        pdf.savefig(fig1)
        plt.close(fig1)

        #PAGE 2: Latency per Frame Plot
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(times)
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency per Frame")
        ax2.axhline(100, linestyle="--")
        plt.tight_layout()

        pdf.savefig(fig2)
        plt.close(fig2)

    print(f"\nLatency report saved to {output_path}")

def main():
    frames_to_test = 200
    print("Starting latency test...")

    #Measure latency
    times = measure_live_latency(recognizer, frames=frames_to_test)

    #Compute stats
    avg = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"\nAverage latency: {avg:.2f} ms")
    print(f"Min: {min_time:.2f} ms")
    print(f"Max: {max_time:.2f} ms")
    print("Target met:", "YES" if avg <= 100 else "NO")

    #Save to PDF
    save_latency_report(times, avg, min_time, max_time, LATENCY_PDF)

if __name__ == "__main__":
    main()
