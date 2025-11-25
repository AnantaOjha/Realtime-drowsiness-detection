# 🚗 Realtime Driver Drowsiness Detector (MediaPipe FaceMesh)

A realtime AI-based driver monitoring system built with **MediaPipe FaceMesh** and **MediaPipe Hands**.  
This application detects **eye closure**, **yawning**, **mouth movement**, and **phone-to-ear behavior** using webcam video input.  
It triggers alarms, overlays alerts, and even supports a **1-minute emergency talk mode**.

---

## 🔥 Features

### 👁️ **Drowsiness Detection (EAR)**
- Uses **Eye Aspect Ratio (EAR)** to detect eye closure.
- Triggers alarm after configurable consecutive frames.

### 😮‍💨 **Yawn Detection (MAR)**
- Uses **Mouth Aspect Ratio (MAR)** to detect yawning.
- Displays on-screen warnings and optional sound alert.

### 📱 **Phone-to-Ear Detection**
- Uses **MediaPipe Hands** to detect if a hand is near the face.
- Warns user with sound + voice alerts.
- Allows a **60-second emergency mode** (press `E`).

### 🔊 **Audio Alarms & TTS**
- Loud alarm using `winsound` (Windows) or terminal beep fallback.
- Optional **text-to-speech warnings** using `pyttsx3`.

### 📈 **Event Logging (CSV)**
- Logs EAR, MAR, FPS, alerts, phone activity, and more.
- Useful for research, tuning, and analysis.

### 🧪 **Debug Mode**
- `--draw` flag visualizes:
  - Eye landmarks
  - Mouth landmarks
  - Nose points
  - Face mesh
