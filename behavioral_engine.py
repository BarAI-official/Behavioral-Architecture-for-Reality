import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import json
import yaml
from collections import deque
from flask import Flask, request, jsonify
from typing import Deque, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------- Konfiguracja ---------------------------------------------------
class Config:
    def __init__(self, path="barai_config.yaml"):
        self.default = {
            "video_src": 0,
            "frame_width": 640,
            "frame_height": 480,
            "max_seq": 30,
            "landmarks_dim": 468,
            "api_port": 8080,
            "ai_model": "svm"
        }
        self.path = path
        self.load()

    def load(self):
        try:
            with open(self.path, "r") as f:
                data = yaml.safe_load(f)
            self.__dict__.update(data)
        except Exception:
            self.__dict__.update(self.default)

cfg = Config()

# --------- Narzędzia -----------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def normalize(v):
    return (v - np.mean(v)) / (np.std(v) + 1e-6)

# --------- Klasa do obsługi video ----------------------------------------
class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

# --------- Klasa do wykrywania twarzy -------------------------------------
class FaceDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        landmarks = res.multi_face_landmarks[0]
        arr = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float32)
        return arr

# --------- Feature Engineering (mikroekspresje, kąty głowy) --------------
def features_from_landmarks(lm):
    if lm is None or len(lm) < 468:
        return np.zeros(10)
    # Odległości oczu i ust
    eye_left = np.linalg.norm(lm[33][:2] - lm[133][:2])
    eye_right = np.linalg.norm(lm[362][:2] - lm[263][:2])
    mouth = np.linalg.norm(lm[61][:2] - lm[291][:2])
    brow_lift = np.linalg.norm(lm[70][:2] - lm[63][:2])
    brow_squeeze = np.linalg.norm(lm[105][:2] - lm[334][:2])
    head_pitch = lm[10][2] - lm[152][2]
    head_yaw = lm[234][0] - lm[454][0]
    chin = np.linalg.norm(lm[152][:2] - lm[8][:2])
    nose_width = np.linalg.norm(lm[94][:2] - lm[331][:2])
    face_width = np.linalg.norm(lm[234][:2] - lm[454][:2])
    feats = np.array([
        eye_left, eye_right, mouth, brow_lift, brow_squeeze,
        head_pitch, head_yaw, chin, nose_width, face_width
    ], dtype=np.float32)
    return normalize(feats)

# --------- Bufor czasowy --------------------------------------------------
class SequenceBuffer:
    def __init__(self, maxlen=30, dim=10):
        self.buf: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.dim = dim

    def add(self, feat):
        if feat is not None and len(feat) == self.dim:
            self.buf.append(feat)

    def full(self):
        return len(self.buf) == self.buf.maxlen

    def get_sequence(self):
        if self.full():
            return np.stack(self.buf, axis=0)
        else:
            return None

    def clear(self):
        self.buf.clear()

# --------- Prosty model AI (SVM) ------------------------------------------
class SimpleBehaviorModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(probability=True)
        self.trained = False

    def train(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.trained = True

    def predict(self, feat):
        if not self.trained:
            return {"emotion": "neutral", "decision": "none", "score": 0.5}
        Xs = self.scaler.transform([feat])
        proba = self.model.predict_proba(Xs)[0]
        idx = np.argmax(proba)
        labels = self.model.classes_
        return {
            "emotion": labels[idx],
            "decision": "positive" if proba[idx] > 0.7 else "uncertain",
            "score": float(proba[idx])
        }

# --------- REST API (Flask) -----------------------------------------------
app = Flask(__name__)
global_engine = None

@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    feat = np.array(data.get('features'), dtype=np.float32)
    res = global_engine.behavior_model.predict(feat)
    return jsonify(res)

@app.route('/train', methods=['POST'])
def api_train():
    data = request.get_json()
    X = np.array(data['X'], dtype=np.float32)
    y = np.array(data['y'])
    global_engine.behavior_model.train(X, y)
    return jsonify({"status": "trained", "samples": len(y)})

# --------- Gaze Tracking (wektor spojrzenia, pitch/yaw) -------------------
def gaze_vector(lm):
    if lm is None or len(lm) < 468:
        return [0.0, 0.0]
    nose = lm[1][:2]
    left_eye = lm[33][:2]
    right_eye = lm[263][:2]
    gaze_x = (left_eye[0] + right_eye[0]) / 2 - nose[0]
    gaze_y = (left_eye[1] + right_eye[1]) / 2 - nose[1]
    return [float(gaze_x), float(gaze_y)]

# --------- Audio Analysis (dummy, rozbuduj wg potrzeb) --------------------
import sounddevice as sd
import scipy.fftpack

class AudioAnalyzer:
    def __init__(self, rate=16000, duration=1):
        self.rate = rate
        self.duration = duration
        self.audio_feat = np.zeros(10)

    def record(self):
        audio = sd.rec(int(self.duration * self.rate), samplerate=self.rate, channels=1)
        sd.wait()
        audio = audio.flatten()
        return audio

    def extract_features(self, audio):
        fft = np.abs(scipy.fftpack.fft(audio))[:500]
        feat = [
            np.mean(fft),
            np.std(fft),
            np.max(fft),
            np.min(fft),
            np.median(fft),
            np.percentile(fft, 25),
            np.percentile(fft, 75),
            np.argmax(fft),
            np.argmin(fft),
            np.sum(fft)
        ]
        return np.array(feat)

    def analyze(self):
        audio = self.record()
        self.audio_feat = self.extract_features(audio)
        return self.audio_feat

# --------- Klasa główna silnika -------------------------------------------
class BarAIBrain:
    def __init__(self):
        self.vid = VideoStream(cfg.video_src, cfg.frame_width, cfg.frame_height)
        self.detector = FaceDetector()
        self.seqbuf = SequenceBuffer(cfg.max_seq)
        self.behavior_model = SimpleBehaviorModel()
        self.audio = AudioAnalyzer()
        self.lock = threading.Lock()

    def analyze_frame(self, frame):
        lm = self.detector.extract(frame)
        feat = features_from_landmarks(lm)
        gaze = gaze_vector(lm)
        aud = self.audio.analyze()
        all_feats = np.concatenate([feat, gaze, aud])
        self.seqbuf.add(all_feats)
        return all_feats

    def predict(self):
        seq = self.seqbuf.get_sequence()
        if seq is not None:
            avg_feat = np.mean(seq, axis=0)
            res = self.behavior_model.predict(avg_feat)
            return res
        else:
            return {"emotion": "neutral", "decision": "none", "score": 0.5}

    def run_live(self):
        self.vid.start()
        try:
            while True:
                frame = self.vid.read()
                if frame is not None:
                    feats = self.analyze_frame(frame)
                    res = self.predict()
                    txt = f"Emotion: {res['emotion']} | Decision: {res['decision']} | Score: {res['score']:.2f}"
                    cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("BarAI Live Analysis", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.vid.stop()
            cv2.destroyAllWindows()

    def save_sequence(self, path="sequence.json"):
        seq = self.seqbuf.get_sequence()
        if seq is not None:
            with open(path, "w") as f:
                json.dump(seq.tolist(), f)

# --------- Main CLI -------------------------------------------------------
def main():
    global global_engine
    global_engine = BarAIBrain()
    log("Starting BarAI behavioral engine...")
    threading.Thread(target=lambda: app.run(port=cfg.api_port, debug=False), daemon=True).start()
    global_engine.run_live()

if __name__ == "__main__":
    main()
