import os
import time
import threading
import tkinter as tk

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk, ImageGrab

# Thumbs down
# Victory (Peace Sign)
# Thumbs up
# Pointing
# Fist Closed
# Open palm

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
screenshot_cooldown = 2.0  # seconds between automatic screenshots


# ----------------------------
# Function to annotate the frame with gesture and landmarks
# ----------------------------
def annotate_frame(frame, top_gesture, hand_landmarks):
    h, w, _ = frame.shape

    # Display the recognized gesture
    if top_gesture:
        text = f"Gesture: {top_gesture.category_name} ({top_gesture.score:.2f})"
        cv2.putText(frame, text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    global screenshot_on_gesture, last_screenshot_time

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

        # Annotate the ORIGINAL BGR frame (so OpenCV drawing colors look right)
        annotate_frame(local_frame, top_gesture, hand_landmarks)

        with lock:
            processed_frame = local_frame

        # If screenshot-on-gesture is enabled and a gesture was detected,
        # capture the screen (rate-limited) and save it to the dataset folder.
        if screenshot_on_gesture and top_gesture:
            now = time.time()
            if now - last_screenshot_time >= screenshot_cooldown:
                try:
                    img = ImageGrab.grab()
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    gesture_name = top_gesture.category_name.replace(" ", "_") if top_gesture.category_name else "gesture"
                    filename = os.path.join(screenshot_folder, f"screenshot_{gesture_name}_{ts}.png")
                    img.save(filename)
                    print(f"Saved screenshot: {filename}")
                    last_screenshot_time = now
                except Exception as e:
                    print(f"Failed to take screenshot: {e}")
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

    # Call update_frame again after 10 ms
    label_img.after(10, update_frame)


# ----------------------------
# Function to start recognition
# ----------------------------
def start_recognition():
    global recognition_running
    if recognition_running:
        return

    recognition_running = True
    start_button.config(state=tk.DISABLED, bg="#4CAF50", fg="white")
    stop_button.config(state=tk.NORMAL, bg="#F44336", fg="white")

    # Start a separate thread for gesture processing
    threading.Thread(target=process_gestures, daemon=True).start()


def toggle_screenshot():
    global screenshot_on_gesture
    screenshot_on_gesture = not screenshot_on_gesture
    if screenshot_on_gesture:
        screenshot_button.config(text="Screenshot: ON", bg="#FFEB3B")
    else:
        screenshot_button.config(text="Screenshot: OFF", bg="#9E9E9E")


# ----------------------------
# Function to stop recognition
# ----------------------------
def stop_recognition():
    global recognition_running
    recognition_running = False
    stop_button.config(state=tk.DISABLED, bg="#9E9E9E", fg="black")
    start_button.config(state=tk.NORMAL, bg="#4CAF50", fg="white")


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
root.title("Hand Gesture Recognition")
root.geometry("800x600")
root.configure(bg="#2196F3")

label_img = tk.Label(root, bg="#2196F3")
label_img.pack(pady=10)

btn_frame = tk.Frame(root, bg="#2196F3")
btn_frame.pack(pady=10)

start_button = tk.Button(btn_frame, text="Start", command=start_recognition,
                         width=12, bg="#4CAF50", fg="white")
start_button.grid(row=0, column=0, padx=8)

stop_button = tk.Button(btn_frame, text="Stop", command=stop_recognition,
                        width=12, state=tk.DISABLED, bg="#9E9E9E")
stop_button.grid(row=0, column=1, padx=8)

screenshot_button = tk.Button(btn_frame, text="Screenshot: OFF", command=toggle_screenshot,
                              width=14, bg="#9E9E9E")
screenshot_button.grid(row=0, column=2, padx=8)

exit_button = tk.Button(btn_frame, text="Exit", command=exit_app, width=12)
exit_button.grid(row=0, column=3, padx=8)

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
