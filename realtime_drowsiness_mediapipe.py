"""Realtime drowsiness detector using MediaPipe FaceMesh.

This file was cleaned and refactored: single entrypoint, drawing of landmarks,
and a debug `--draw` flag to visualize eye/mouth/nose points.
"""

from collections import deque
import argparse
import math
import time
import cv2
import numpy as np
import winsound
import threading
# pyttsx3 is optional at runtime; fallback to console printing if unavailable
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    _TTS_AVAILABLE = False
import csv
import os
# use `deque` imported above; avoid duplicate imports that trigger linter warnings

try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe is required. Run `pip install mediapipe`") from e

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Example landmark indices for eyes/mouth/nose (MediaPipe FaceMesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [13, 14, 78, 308]  # outer lip-ish points
NOSE_TIP_IDX = [1, 4]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ear(landmarks, idxs, img_w, img_h):
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idxs]
    p1, p2, p3, p4, p5, p6 = pts
    ear = (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)
    return ear, pts


def compute_mar(landmarks, img_w, img_h):
    # Use outer lip top (13) and bottom (14) and left (78) right (308)
    try:
        top = (int(landmarks[13].x * img_w), int(landmarks[13].y * img_h))
        bottom = (int(landmarks[14].x * img_w), int(landmarks[14].y * img_h))
        left = (int(landmarks[78].x * img_w), int(landmarks[78].y * img_h))
        right = (int(landmarks[308].x * img_w), int(landmarks[308].y * img_h))
        v = dist(top, bottom)
        h = dist(left, right) + 1e-6
        mar = v / h
        return mar, (top, bottom, left, right)
    except Exception:
        return 0.0, []


def draw_points(frame, pts, color=(0, 255, 0), radius=2):
    for (x, y) in pts:
        cv2.circle(frame, (x, y), radius, color, -1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--ear-thresh", type=float, default=0.22)
    p.add_argument("--ear-frames", type=int, default=15, help="Consecutive frames for eye close alert")
    p.add_argument("--mar-thresh", type=float, default=0.6, help="MAR threshold for yawning (ratio)")
    p.add_argument("--mar-frames", type=int, default=20, help="Consecutive frames for yawn alert")
    p.add_argument("--smooth-windows", type=int, default=5)
    p.add_argument("--draw", action="store_true", help="Draw landmarks and debug points")
    p.add_argument("--log", action="store_true", help="Enable CSV logging of frame metrics and events")
    p.add_argument("--model-complexity", type=int, default=0, choices=[0,1,2], help="FaceMesh model complexity (0-2)")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Unable to open camera {args.camera}")
        return

    ear_deque = deque(maxlen=args.smooth_windows)
    consec_eye = 0
    consec_yawn = 0
    alert_eye = False
    alert_yawn = False
    alarm_active = False

    # Alarm thread control
    alarm_thread = None
    alarm_stop_event = threading.Event()

    def alarm_worker(stop_event: threading.Event):
        # Looping beep until stop_event is set
        try:
            while not stop_event.is_set():
                try:
                    # frequency 1500 Hz for 400ms
                    winsound.Beep(1500, 400)
                except Exception:
                    # fallback to system sound if Beep not available
                    try:
                        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
                    except Exception:
                        pass
                # small delay, allow responsive stop
                if stop_event.wait(0.1):
                    break
        except Exception:
            return

    def start_alarm():
        nonlocal alarm_thread, alarm_stop_event
        if alarm_thread and alarm_thread.is_alive():
            return
        alarm_stop_event.clear()
        alarm_thread = threading.Thread(target=alarm_worker, args=(alarm_stop_event,), daemon=True)
        alarm_thread.start()
        nonlocal alarm_active
        alarm_active = True

    def stop_alarm():
        nonlocal alarm_thread, alarm_stop_event
        try:
            alarm_stop_event.set()
        except Exception:
            pass
        alarm_thread = None
        nonlocal alarm_active
        alarm_active = False

    # TTS helper (runs in background thread to avoid blocking)
    def speak(text: str):
        def _s():
            if _TTS_AVAILABLE:
                try:
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    return
            else:
                # fallback: print to console so user still receives the message
                try:
                    print("[TTS]", text)
                except Exception:
                    pass
        t = threading.Thread(target=_s, daemon=True)
        t.start()

    # Hands detector (to detect hand near face -> likely phone)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    phone_near = False
    emergency_on = False
    emergency_end_time = 0.0
    phone_warning_sent = False
    phone_warning_time = 0.0
    PHONE_WARNING_COOLDOWN = 8.0  # seconds between repeated warnings
    MOUTH_OPEN_THRESH = 0.03  # relative to face height

    # optional refinement is more expensive; use model_complexity from args
    refine = False
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=refine,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        try:
            # logging setup
            log_fp = None
            csv_writer = None
            if args.log:
                os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
                log_fp = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'events_log.csv'), 'a', newline='')
                csv_writer = csv.writer(log_fp)
                # write header if file is empty
                if os.stat(log_fp.fileno()).st_size == 0:
                    csv_writer.writerow(['timestamp', 'ear', 'mar', 'alert_eye', 'alert_yawn', 'phone_near', 'mouth_open', 'fps'])

            # performance tracking
            proc_times = deque(maxlen=30)
            fps = 0.0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: empty frame")
                    break
                start = time.time()
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                hands_res = hands.process(rgb)

                if results.multi_face_landmarks:
                    fa = results.multi_face_landmarks[0]

                    if args.draw:
                        mp_draw.draw_landmarks(frame, fa, mp_face.FACEMESH_TESSELATION,
                                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                               mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1))

                    landmarks = fa.landmark

                    # compute face bbox size and center for proximity checks
                    xs = [l.x for l in landmarks]
                    ys = [l.y for l in landmarks]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    face_w = (max_x - min_x) * w
                    face_h = (max_y - min_y) * h
                    face_cx = int(((min_x + max_x) / 2.0) * w)
                    face_cy = int(((min_y + max_y) / 2.0) * h)

                    # compute EAR and draw points
                    try:
                        left_ear, left_pts = compute_ear(landmarks, LEFT_EYE_IDX, w, h)
                        right_ear, right_pts = compute_ear(landmarks, RIGHT_EYE_IDX, w, h)
                        ear = (left_ear + right_ear) / 2.0
                        ear_deque.append(ear)
                        ear_smooth = sum(ear_deque) / len(ear_deque)

                        # MAR (yawn) computation
                        mar, mar_pts = compute_mar(landmarks, w, h)
                        if args.draw and mar_pts:
                            # draw mouth mar points
                            for p in mar_pts:
                                if isinstance(p, tuple):
                                    cv2.circle(frame, p, 2, (0,255,255), -1)

                        cv2.putText(frame, f"EAR:{ear_smooth:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)

                        if args.draw:
                            draw_points(frame, left_pts, (255, 0, 0), 2)
                            draw_points(frame, right_pts, (255, 0, 0), 2)

                        # Eye consecutive logic
                        if ear_smooth < args.ear_thresh:
                            consec_eye += 1
                        else:
                            if consec_eye >= args.ear_frames:
                                print(f"Drowsiness event detected (frames={consec_eye})")
                                # log event
                                if csv_writer:
                                    csv_writer.writerow([time.time(), f"{ear_smooth:.4f}", f"{mar:.4f}", 1, 0, int(phone_near), int(mouth_open), f"{fps:.2f}"])
                            # Eyes opened — ensure any running alarm is stopped
                            if alert_eye or alarm_active:
                                stop_alarm()
                            alert_eye = False
                            consec_eye = 0

                        if consec_eye >= args.ear_frames:
                            if not alert_eye:
                                alert_eye = True
                                start_alarm()
                            # draw translucent red overlay to indicate alert
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                            cv2.putText(frame, "DROWSINESS ALERT!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (255, 255, 255), 3)

                        # MAR (yawn) consecutive logic
                        if mar > args.mar_thresh:
                            consec_yawn += 1
                        else:
                            if consec_yawn >= args.mar_frames:
                                print(f"Yawn event detected (frames={consec_yawn})")
                                if csv_writer:
                                    csv_writer.writerow([time.time(), f"{ear_smooth:.4f}", f"{mar:.4f}", 0, 1, int(phone_near), int(mouth_open), f"{fps:.2f}"])
                            consec_yawn = 0
                        if consec_yawn >= args.mar_frames:
                            if not alert_yawn:
                                alert_yawn = True
                                # Yawn alert: short beep + overlay
                                try:
                                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
                                except Exception:
                                    pass
                            cv2.putText(frame, "YAWN DETECTED", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (0, 255, 255), 3)
                    except Exception:
                        # If any landmark indexing fails, continue and show all landmarks for debugging
                        if args.draw:
                            cv2.putText(frame, "LANDMARK INDEX ERROR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2)

                    # detect if a hand is near the face (phone pick-up approximation)
                    phone_near = False
                    if hands_res and hands_res.multi_hand_landmarks:
                        for hand_landmarks in hands_res.multi_hand_landmarks:
                            # use wrist (landmark 0) as representative point
                            hx = int(hand_landmarks.landmark[0].x * w)
                            hy = int(hand_landmarks.landmark[0].y * h)
                            d = dist((hx, hy), (face_cx, face_cy))
                            # if wrist is within ~0.6 * face_width, consider hand near face
                            if d < max(40, 0.6 * max(face_w, face_h)):
                                phone_near = True
                                break

                    # detect mouth opening (talking proxy)
                    mouth_open = False
                    try:
                        top_lip = landmarks[13]
                        bottom_lip = landmarks[14]
                        mouth_v = abs(top_lip.y - bottom_lip.y) * h
                        if face_h > 0:
                            if (mouth_v / face_h) > MOUTH_OPEN_THRESH:
                                mouth_open = True
                    except Exception:
                        mouth_open = False

                    # draw mouth and nose debug points
                    if args.draw:
                        try:
                            mouth_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_IDX]
                            nose_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in NOSE_TIP_IDX]
                            draw_points(frame, mouth_pts, (0, 255, 255), 3)
                            draw_points(frame, nose_pts, (0, 165, 255), 3)
                            cv2.putText(frame, "MOUTH/Nose points shown", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 255), 1)
                        except Exception:
                            pass
                    # when face present, show phone warning if hand near
                    if phone_near:
                        cv2.putText(frame, "PHONE TO EAR DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 165, 255), 2)
                        cv2.putText(frame, "Press 'e' to allow 1-min emergency", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1)
                        # Speak a warning once per cooldown when phone is first detected
                        now = time.time()
                        if not phone_warning_sent or (now - phone_warning_time) > PHONE_WARNING_COOLDOWN:
                            phone_warning_sent = True
                            phone_warning_time = now
                            speak("Warning. Do not talk while driving. If it is an emergency, press E to allow one minute of speaking.")
                        # if mouth open and not emergency, remind user
                        if mouth_open and not emergency_on:
                            speak("Please do not talk. This is not safe. Press E only for emergencies.")
                        # if not emergency, ensure alarm running
                        if not emergency_on:
                            if not alarm_active:
                                alarm_active = True
                                start_alarm()
                    # handle emergency timeout
                    if emergency_on and time.time() >= emergency_end_time:
                        emergency_on = False
                        speak("Emergency time ended")
                        # if phone still near, resume alarm
                        if phone_near:
                            if not alarm_active:
                                alarm_active = True
                                start_alarm()
                    # reset phone warning when phone is removed
                    if not phone_near:
                        phone_warning_sent = False
                else:
                    if args.draw:
                        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 255), 2)

                # performance: compute frame proc time and fps
                proc_time = time.time() - start
                proc_times.append(proc_time)
                if len(proc_times) > 0:
                    fps = 1.0 / (sum(proc_times) / len(proc_times))
                cv2.putText(frame, f"FPS:{fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                cv2.putText(frame, f"Proc:{proc_time*1000:.0f}ms", (w - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                cv2.imshow("Drowsiness", frame)
                # log every frame if logging enabled
                if csv_writer:
                    try:
                        csv_writer.writerow([time.time(), f"{ear_smooth:.4f}", f"{mar:.4f}", int(alert_eye), int(alert_yawn), int(phone_near), int(mouth_open), f"{fps:.2f}"])
                    except Exception:
                        pass
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('e'):
                    # start emergency talk allowance for 60 seconds
                    emergency_on = True
                    emergency_end_time = time.time() + 60.0
                    # stop alarm while emergency allowed
                    if alarm_active:
                        alarm_active = False
                        stop_alarm()
                    speak("Emergency mode enabled. You may speak for one minute.")
        finally:
            cap.release()
            try:
                hands.close()
            except Exception:
                pass
            if args.log and log_fp:
                try:
                    log_fp.close()
                except Exception:
                    pass
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
