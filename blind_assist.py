# yolo_tts_distance_calibrated.py

from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import queue
import time

# ======================
#  TTS WORKER THREAD
# ======================

def tts_worker(q):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = q.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

# ======================
#  DETECTION HELPERS
# ======================

def extract_box_data(box, names):
    """
    Extract bounding box and meta data from a YOLO box.
    No NumPy used, just tensor -> Python types.
    """
    xyxy = box.xyxy[0]  # tensor of [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, xyxy.tolist())

    conf = float(box.conf[0])
    cls = int(box.cls[0])

    # names is usually a dict: {class_id: "label"}
    name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)

    height = y2 - y1
    return x1, y1, x2, y2, conf, cls, name, height

# ======================
#  CALIBRATION-BASED DISTANCE
# ======================

# Calibration constants (example values)
# distance (meters) ≈ calibration_constant / bbox_height_pixels
#
# You should measure real (bbox_height, distance) pairs with your camera setup
# and update these values.
CALIBRATION_DEFAULT = 460.0  # fallback for unknown classes

CALIBRATION_TABLE = {
    "person": 460.0,   # example value for person
    # "chair": 300.0,  # you can add per-class constants later if you want
}

def estimate_distance_for_class(bbox_height, name):
    """
    Uses a calibration constant for each class name (if available).
    distance ≈ k / bbox_height

    bbox_height: height of the detection box in pixels
    name: class label, e.g. "person"
    """
    if bbox_height <= 0:
        return float('inf')

    k = CALIBRATION_TABLE.get(name, CALIBRATION_DEFAULT)
    distance = k / float(bbox_height)  # in meters if k is calibrated with meters
    return distance

# ======================
#  DESCRIPTION FOR TTS
# ======================

def build_description(detections, frame_width):
    """
    detections: list of (x1, y1, x2, y2, conf, cls, name, distance)
    frame_width: width of the current frame in pixels
    """
    if not detections:
        return "No objects detected."

    # remove infinite distances
    detections = [d for d in detections if d[7] != float('inf')]
    if not detections:
        return "No valid objects detected."

    # sort nearest first
    detections = sorted(detections, key=lambda d: d[7])

    max_objects = 3  # describe at most 3 objects
    parts = []

    for (x1, y1, x2, y2, conf, cls, name, distance) in detections[:max_objects]:
        center_x = (x1 + x2) // 2

        # side: left / right / center
        if center_x < frame_width / 3:
            side = "on your left"
        elif center_x > 2 * frame_width / 3:
            side = "on your right"
        else:
            side = "in front of you"

        # distance wording: <=5 -> very close, >5 -> nearby
        if distance <= 5:
            zone = "very close"
        else:
            zone = "nearby"

        parts.append(
            f"{name} {zone} {side}, distance {distance:.1f} meters"
        )

    return ". ".join(parts)

# ======================
#  MAIN LOOP
# ======================

def main():
    # TTS thread
    tts_queue = queue.Queue()
    t = threading.Thread(target=tts_worker, args=(tts_queue,), daemon=True)
    t.start()

    # Load YOLO model
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Lower resolution for performance if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_detections = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_width = w

            # YOLO inference
            results = model.predict(
                frame,
                imgsz=416,    # smaller than 640 for speed
                conf=0.35,
                verbose=False
            )[0]

            detected = []
            if results.boxes is not None and len(results.boxes) > 0:
                names = results.names
                for box in results.boxes:
                    x1, y1, x2, y2, conf, cls, name, height = extract_box_data(box, names)
                    distance = estimate_distance_for_class(height, name)
                    detected.append((x1, y1, x2, y2, conf, cls, name, distance))

            last_detections = detected

            # Draw boxes and distance
            for x1, y1, x2, y2, conf, cls, name, distance in detected:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"{name} {distance:.1f}m",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("YOLO Distance TTS (calibrated, on-demand)", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # quit
                break

            elif key == ord('b'):
                # button: describe scene
                text = build_description(last_detections, frame_width)
                print("TTS:", text)
                tts_queue.put(text)

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put(None)
        t.join()

# ======================
#  ENTRY POINT
# ======================

if __name__ == "__main__":
    main()
