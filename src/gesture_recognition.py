import os
import time
import threading
import tkinter as tk

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk


# ----------------------------
# UI color palette (dark theme)
# ----------------------------
C_BG      = "#0F0F17"   # main background
C_PANEL   = "#1A1A27"   # card / panel surfaces
C_ACCENT  = "#6366F1"   # indigo accent (camera border)
C_SUCCESS = "#22C55E"   # green  – Start button / running indicator
C_DANGER  = "#EF4444"   # red    – Stop (active) / Exit
C_WARN    = "#F59E0B"   # amber  – Auto-Capture ON
C_MUTED   = "#374151"   # grey   – disabled state
C_TEXT    = "#F1F5F9"   # primary text
C_SUBTEXT = "#94A3B8"   # secondary / status text


# Directory to store the screenshots
screenshot_folder = "Dataset"

# Create the directory if it doesn't exist
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

# ----------------------------
# Initialize GestureRecognizer object
# ----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "gesture_recognizer.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model file:\n{MODEL_PATH}\n\n"
        "Place gesture_recognizer.task in the same folder as gesture_recognition.py"
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Global variables
recognition_running = False
frame_to_process = None
processed_frame = None
lock = threading.Lock()

# Screenshot-on-gesture toggle and rate limiting
screenshot_on_gesture = False
last_screenshot_time = 0.0
screenshot_cooldown = 5.0  # seconds between automatic screenshots
current_gesture_text = "—  No gesture detected"  # shown in the UI gesture label


# ----------------------------
# Function to annotate the frame with gesture and landmarks
# ----------------------------
def annotate_frame(frame, hand_landmarks):
    h, w, _ = frame.shape

    # Draw hand landmarks
    if hand_landmarks:
        for landmark_set in hand_landmarks:
            for lm in landmark_set:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)


# ----------------------------
# Function to process the frames for gesture recognition 
# ----------------------------
def process_gestures():
    global frame_to_process, processed_frame, recognition_running
    global screenshot_on_gesture, last_screenshot_time, current_gesture_text

    while recognition_running:
        local_frame = None
        with lock:
            if frame_to_process is not None:
                local_frame = frame_to_process.copy()

        if local_frame is None:
            time.sleep(0.005)
            continue

        # Convert frame to MediaPipe image format
        frame_rgb_for_mp = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_for_mp)

        # Recognize gesture
        recognition_result = recognizer.recognize(mp_image)

        top_gesture = recognition_result.gestures[0][0] if recognition_result.gestures else None
        hand_landmarks = recognition_result.hand_landmarks if recognition_result.hand_landmarks else []

        # Save a clean copy before drawing landmarks (cleaner for dataset use)
        save_frame = local_frame.copy()

        # Annotate the ORIGINAL BGR frame (so OpenCV drawing colors look right)
        annotate_frame(local_frame, hand_landmarks)

        # Update gesture status for the UI label (read by the main thread in update_frame)
        if top_gesture:
            current_gesture_text = (
                f"✋  {top_gesture.category_name}"
                f"   —   {top_gesture.score:.0%} confidence"
            )
        else:
            current_gesture_text = "—  No gesture detected"

        with lock:
            processed_frame = local_frame

        # If screenshot-on-gesture is enabled and a gesture was detected,
        # capture the screen (rate-limited) and save it to the dataset folder.
        if screenshot_on_gesture and top_gesture:
            now = time.time()
            if now - last_screenshot_time >= screenshot_cooldown:
                try:
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    gesture_name = top_gesture.category_name.replace(" ", "_") if top_gesture.category_name else "gesture"
                    filename = os.path.join(screenshot_folder, f"{gesture_name}_{ts}.png")
                    cv2.imwrite(filename, save_frame)
                    print(f"Saved frame: {filename}")
                    last_screenshot_time = now
                except Exception as e:
                    print(f"Failed to save frame: {e}")
        time.sleep(0.005)


# ----------------------------
# Function to capture frames and update Tkinter window
# ----------------------------
def update_frame():
    global frame_to_process, processed_frame

    ret, frame = cap.read()
    if not ret:
        print("Unable to retrieve frame. Exiting ...")
        exit_app()
        return

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    with lock:
        frame_to_process = frame

    with lock:
        display_frame = processed_frame if processed_frame is not None else frame

    # Convert BGR -> RGB for Tkinter
    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

    # Update the image on the Tkinter label
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)

    # Sync gesture status label from background thread's last result
    gesture_var.set(current_gesture_text)

    # Call update_frame again after 10 ms
    label_img.after(10, update_frame)


def toggle_recognition():
    global recognition_running
    if recognition_running:
        # ── Stop ──
        recognition_running = False
        toggle_button.config(text="▶  Start", bg=C_SUCCESS, fg=C_TEXT)
        status_bar.config(text="●  Stopped", fg=C_SUBTEXT)
        gesture_var.set("—  No gesture detected")
    else:
        # ── Start ──
        recognition_running = True
        toggle_button.config(text="■  Stop", bg=C_DANGER, fg=C_TEXT)
        if screenshot_on_gesture:
            _update_capture_countdown()
        else:
            status_bar.config(text="●  Recognition running", fg=C_SUCCESS)
        threading.Thread(target=process_gestures, daemon=True).start()


def toggle_screenshot():
    global screenshot_on_gesture, last_screenshot_time
    screenshot_on_gesture = not screenshot_on_gesture
    if screenshot_on_gesture:
        last_screenshot_time = time.time()  # start the cooldown fresh when enabled
        screenshot_button.config(text="◉  Auto-Capture: ON", bg=C_WARN, fg="#0F0F17")
        if recognition_running:
            _update_capture_countdown()
    else:
        screenshot_button.config(text="◉  Auto-Capture: OFF", bg=C_MUTED, fg=C_TEXT)
        if recognition_running:
            status_bar.config(text="●  Recognition running", fg=C_SUCCESS)


def _update_capture_countdown():
    """Tick every 500 ms and show remaining seconds until the next auto-capture."""
    if not (screenshot_on_gesture and recognition_running):
        return  # one of the conditions dropped — let the ticker die naturally
    remaining = max(0.0, screenshot_cooldown - (time.time() - last_screenshot_time))
    status_bar.config(text=f"◉  Next capture in  {remaining:.0f}s", fg=C_WARN)
    root.after(500, _update_capture_countdown)


# ----------------------------
# Function to exit recognition
# ----------------------------
def exit_app():
    global recognition_running
    recognition_running = False
    try:
        cap.release()
    except Exception:
        pass
    root.quit()

# ----------------------------
# Initialize the Tkinter window
# ----------------------------
root = tk.Tk()
root.title("Hand Gesture Recognizer")
root.resizable(False, False)
root.configure(bg=C_BG)

# ── Header ───────────────────────────────────────────────────────────
header = tk.Frame(root, bg=C_BG)
header.pack(fill=tk.X, padx=28, pady=(20, 14))

tk.Label(header, text="Hand Gesture Recognizer",
         font=("Helvetica", 18, "bold"), bg=C_BG, fg=C_TEXT).pack()
tk.Label(header, text="Real-time detection via MediaPipe",
         font=("Helvetica", 10), bg=C_BG, fg=C_SUBTEXT).pack()

# ── Camera view (accent border → dark inner frame → live feed) ────────
cam_border = tk.Frame(root, bg=C_ACCENT, padx=2, pady=2)
cam_border.pack(padx=28)

cam_inner = tk.Frame(cam_border, bg=C_PANEL)
cam_inner.pack()

label_img = tk.Label(cam_inner, bg=C_PANEL)
label_img.pack()

# ── Gesture status label ──────────────────────────────────────────────
gesture_var = tk.StringVar(value="—  No gesture detected")

gesture_label = tk.Label(
    root,
    textvariable=gesture_var,
    font=("Helvetica", 17, "bold"),
    bg=C_PANEL, fg=C_TEXT,
    anchor="center",
    padx=0, pady=16,
)
gesture_label.pack(fill=tk.X, padx=28, pady=(10, 0))

# ── Control buttons ───────────────────────────────────────────────────
btn_frame = tk.Frame(root, bg=C_BG)
btn_frame.pack(pady=16)


def _btn(parent, text, cmd, bg, fg=C_TEXT, width=16):
    # Use tk.Label instead of tk.Button — on macOS, tk.Button uses native Aqua
    # rendering that ignores bg/fg entirely. tk.Label honours all colour settings.
    b = tk.Label(
        parent, text=text,
        font=("Helvetica", 10, "bold"),
        width=width, bg=bg, fg=fg,
        padx=10, pady=8,
        cursor="hand2",
        anchor="center",
        relief=tk.FLAT,
    )
    b.bind("<Button-1>", lambda *_: cmd())
    b.bind("<Enter>",    lambda *_: b.config(relief=tk.GROOVE))
    b.bind("<Leave>",    lambda *_: b.config(relief=tk.FLAT))
    return b


toggle_button     = _btn(btn_frame, "▶  Start",             toggle_recognition, C_SUCCESS)
screenshot_button = _btn(btn_frame, "◉  Auto-Capture: OFF", toggle_screenshot,  C_MUTED, width=20)
exit_button       = _btn(btn_frame, "✕  Exit",              exit_app,           C_DANGER)

toggle_button.grid    (row=0, column=0, padx=6)
screenshot_button.grid(row=0, column=1, padx=6)
exit_button.grid      (row=0, column=2, padx=6)

# ── Status bar ────────────────────────────────────────────────────────
status_bar = tk.Label(
    root,
    text="●  Ready",
    font=("Helvetica", 9),
    bg="#09090F", fg=C_SUBTEXT,
    anchor="w", padx=14, pady=5,
)
status_bar.pack(fill=tk.X, side=tk.BOTTOM)

# ----------------------------
# Camera init + run
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera.\n"
        "On macOS: System Settings → Privacy & Security → Camera → allow Terminal/VS Code/PyCharm."
    )

update_frame()
root.mainloop()
