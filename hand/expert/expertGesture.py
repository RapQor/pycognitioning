import os
import cv2
import math
import time
import argparse
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ML
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# MediaPipe
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----------------------------
# Config & Utilities
# ----------------------------
GESTURE_LABELS_EXAMPLE = [
    "open_palm", "fist", "thumbs_up", "thumbs_down",
    "peace", "ok", "rock", "call_me"
]

def angle_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-9) -> float:
    """Return angle in radians between vectors v1 and v2."""
    dot = float(np.dot(v1, v2))
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + eps
    # clamp to avoid numeric issues
    cosang = max(-1.0, min(1.0, dot / norm))
    return math.acos(cosang)

def tri_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b formed by a-b-c (in radians)."""
    v1 = a - b
    v2 = c - b
    return angle_between(v1, v2)

def ema(prev: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    return current if prev is None else alpha * current + (1 - alpha) * prev

def handedness_onehot(label: str) -> List[float]:
    # label comes like "Right" or "Left"
    if label.lower().startswith("r"):
        return [1.0, 0.0]  # Right, Left
    else:
        return [0.0, 1.0]

# ----------------------------
# Feature Extractor
# ----------------------------
@dataclass
class FeatureConfig:
    include_angles: bool = True
    include_distances: bool = True
    include_handedness: bool = True
    normalize_scale: bool = True
    use_z: bool = False  # set True if you want z features too

class FeatureExtractor:
    """
    Convert 21 landmarks (x,y[,z]) into a robust, scale/translation-invariant feature vector:
    - normalize to wrist origin
    - scale by palm size (wrist to middle_mcp)
    - angles at key joints (rotation tolerance)
    - distance ratios from tips to palm center
    - handedness one-hot
    """
    # Mediapipe landmark indices
    WRIST = 0
    # MCP joints
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    def __init__(self, cfg: FeatureConfig = FeatureConfig()):
        self.cfg = cfg

    def _lm_to_np(self, hand_landmarks) -> np.ndarray:
        if self.cfg.use_z:
            pts = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)
        else:
            pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
        return pts

    def _normalize(self, pts: np.ndarray) -> Tuple[np.ndarray, float]:
        # translate so wrist at origin
        wrist = pts[self.WRIST].copy()
        pts_norm = pts - wrist
        # scale by palm/middle MCP distance -> robust to zoom
        scale_ref = np.linalg.norm(pts[self.MIDDLE_MCP] - pts[self.WRIST]) + 1e-9
        if self.cfg.normalize_scale:
            pts_norm = pts_norm / scale_ref
        return pts_norm, scale_ref

    def _palm_center(self, pts_norm: np.ndarray) -> np.ndarray:
        # average of wrist + all MCP joints as simple palm center
        idx = [self.WRIST, self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP, self.THUMB_CMC]
        return np.mean(pts_norm[idx], axis=0)

    def _finger_joint_angles(self, pts: np.ndarray) -> List[float]:
        # Angles (in radians) at MCP/PIP/DIP for index, middle, ring, pinky
        chains = [
            (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            (self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        ]
        angles = []
        for mcp, pip, dip, tip in chains:
            # angle at MCP using wrist as previous (wrist - mcp - pip)
            angles.append(tri_angle(pts[self.WRIST], pts[mcp], pts[pip]))
            # angle at PIP (mcp - pip - dip)
            angles.append(tri_angle(pts[mcp], pts[pip], pts[dip]))
            # angle at DIP (pip - dip - tip)
            angles.append(tri_angle(pts[pip], pts[dip], pts[tip]))
        # Thumb angles (cmc-mcp-ip-tip chain)
        angles.append(tri_angle(pts[self.WRIST], pts[self.THUMB_MCP], pts[self.THUMB_IP]))
        angles.append(tri_angle(pts[self.THUMB_MCP], pts[self.THUMB_IP], pts[self.THUMB_TIP]))
        return angles  # length = 14

    def _tip_dist_ratios(self, pts_norm: np.ndarray) -> List[float]:
        tips = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        palm = self._palm_center(pts_norm)
        dists = [np.linalg.norm(pts_norm[t] - palm) for t in tips]
        # normalize to sum=1 to make ratios (robust)
        s = sum(dists) + 1e-9
        return [d / s for d in dists]  # length = 5

    def _pairwise_key_dists(self, pts_norm: np.ndarray) -> List[float]:
        # A compact set of pairwise distances useful for shape: tips to their MCPs and neighbors
        pairs = [
            (self.INDEX_TIP, self.INDEX_MCP),
            (self.MIDDLE_TIP, self.MIDDLE_MCP),
            (self.RING_TIP, self.RING_MCP),
            (self.PINKY_TIP, self.PINKY_MCP),
            (self.THUMB_TIP, self.THUMB_CMC),
            (self.INDEX_TIP, self.MIDDLE_TIP),
            (self.MIDDLE_TIP, self.RING_TIP),
            (self.RING_TIP, self.PINKY_TIP),
            (self.THUMB_TIP, self.INDEX_TIP),
        ]
        return [float(np.linalg.norm(pts_norm[a] - pts_norm[b])) for a, b in pairs]  # length = 9

    def extract(self, hand_landmarks, handed_label: str) -> List[float]:
        pts = self._lm_to_np(hand_landmarks)        # (21, 2 or 3)
        pts_norm, _ = self._normalize(pts)          # origin @ wrist, scaled by palm size

        feats: List[float] = []

        if self.cfg.include_angles:
            feats.extend(self._finger_joint_angles(pts))      # 14
        if self.cfg.include_distances:
            feats.extend(self._tip_dist_ratios(pts_norm))     # +5
            feats.extend(self._pairwise_key_dists(pts_norm))  # +9
        # raw normalized landmark subset to help model (compact pose shape)
        key_ids = [self.WRIST, self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP,
                   self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]
        feats.extend(pts_norm[key_ids].flatten().tolist())     # + (10 * dim)

        if self.cfg.include_handedness:
            feats.extend(handedness_onehot(handed_label))      # +2

        return feats

# ----------------------------
# Dataset IO
# ----------------------------
def append_sample_csv(path: str, features: List[float], label: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = ",".join(map(str, features)) + f",{label}\n"
    header_needed = not os.path.exists(path)
    if header_needed:
        # create header with generic f0..fn,label
        with open(path, "w", encoding="utf-8") as f:
            for i in range(len(features)):
                f.write(f"f{i},")
            f.write("label\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

def load_dataset_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    import csv
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            *feat, label = row
            X.append([float(v) for v in feat])
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

# ----------------------------
# Model
# ----------------------------
def train_model(data_csv: str, model_out: str, test_size: float = 0.15, random_state: int = 42):
    X, y = load_dataset_csv(data_csv)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Pipeline: Standardize → MLPClassifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=3e-4,
            max_iter=300,
            random_state=random_state,
            verbose=False
        ))
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"[TRAIN] Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred))

    joblib.dump({
        "pipeline": pipe,
        "labels": sorted(list(set(y)))
    }, model_out)
    print(f"[TRAIN] Model saved to: {model_out}")

# ----------------------------
# Real-time Inference
# ----------------------------
@dataclass
class InferenceConfig:
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2
    prob_ema_alpha: float = 0.4
    vote_window: int = 7
    decision_confidence: float = 0.75  # final threshold

class RealtimeInference:
    def __init__(self, model_path: str, feat_cfg: FeatureConfig = FeatureConfig(), infer_cfg: InferenceConfig = InferenceConfig()):
        bundle = joblib.load(model_path)
        self.pipe: Pipeline = bundle["pipeline"]
        self.labels: List[str] = bundle["labels"]
        self.extractor = FeatureExtractor(feat_cfg)
        self.cfg = infer_cfg

        self.prob_ema_right: Optional[np.ndarray] = None
        self.prob_ema_left: Optional[np.ndarray] = None
        self.votes_right = deque(maxlen=self.cfg.vote_window)
        self.votes_left = deque(maxlen=self.cfg.vote_window)

    def _predict_proba(self, feats: List[float]) -> np.ndarray:
        # sklearn predict_proba
        proba = self.pipe.predict_proba([feats])[0]  # shape (n_classes,)
        # ensure order aligns with pipe.classes_
        return proba

    def _decide(self, ema_probs: np.ndarray) -> Tuple[str, float]:
        idx = int(np.argmax(ema_probs))
        return self.pipe.named_steps["clf"].classes_[idx], float(ema_probs[idx])

    def run(self, cam_index: int = 0):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("[INFER] Cannot open camera")
            return

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.cfg.max_num_hands,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence
        ) as hands:

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                h, w = frame.shape[:2]
                if res.multi_hand_landmarks and res.multi_handedness:
                    for hlms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                        handed_label = handed.classification[0].label  # "Right"/"Left"
                        feats = self.extractor.extract(hlms, handed_label)
                        probs = self._predict_proba(feats)

                        # EMA by hand side
                        if handed_label.lower().startswith("r"):
                            self.prob_ema_right = ema(self.prob_ema_right, probs, self.cfg.prob_ema_alpha)
                            ema_probs = self.prob_ema_right
                            votes = self.votes_right
                        else:
                            self.prob_ema_left = ema(self.prob_ema_left, probs, self.cfg.prob_ema_alpha)
                            ema_probs = self.prob_ema_left
                            votes = self.votes_left

                        pred_label, pred_conf = self._decide(ema_probs)
                        votes.append(pred_label)
                        # majority vote
                        if len(votes) > 0:
                            maj_label = max(set(votes), key=list(votes).count)
                        else:
                            maj_label = pred_label

                        final_label = maj_label if pred_conf >= self.cfg.decision_confidence else "unknown"
                        final_conf = pred_conf if final_label != "unknown" else pred_conf

                        # draw landmarks
                        mp_drawing.draw_landmarks(
                            frame, hlms, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(thickness=2)
                        )

                        # bounding box & text
                        xs = [int(lm.x * w) for lm in hlms.landmark]
                        ys = [int(lm.y * h) for lm in hlms.landmark]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), 2)
                        txt = f"{handed_label}: {final_label} ({final_conf:.2f})"
                        cv2.putText(frame, txt, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # HUD
                cv2.putText(frame, "Infer mode: q=quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, "Infer mode: q=quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                cv2.imshow("Expert Gesture - Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

# ----------------------------
# Data Collection
# ----------------------------
def run_collect(label: str, out_csv: str, cam_index: int = 0):
    extractor = FeatureExtractor()
    cfg = InferenceConfig()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[COLLECT] Cannot open camera")
        return

    print("[COLLECT] Press SPACE to capture sample(s), 'q' to quit.")
    print(f"[COLLECT] Current label: '{label}'")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks and res.multi_handedness:
                for hlms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)
                    handed_label = handed.classification[0].label
                    # simple guide
                    h, w = frame.shape[:2]
                    xs = [int(lm.x * w) for lm in hlms.landmark]
                    ys = [int(lm.y * h) for lm in hlms.landmark]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    cv2.putText(frame, handed_label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"Label: {label} | SPACE=save, q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, f"Label: {label} | SPACE=save, q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("Expert Gesture - Collect", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # SPACE to save
                if res.multi_hand_landmarks and res.multi_handedness:
                    saved = 0
                    for hlms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                        feats = extractor.extract(hlms, handed.classification[0].label)
                        append_sample_csv(out_csv, feats, label)
                        saved += 1
                    print(f"[COLLECT] Saved {saved} sample(s) to {out_csv}")
                else:
                    print("[COLLECT] No hands detected, nothing saved.")
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Expert Static Hand Gesture Recognition")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_collect = sub.add_parser("collect", help="Collect feature samples to CSV")
    p_collect.add_argument("--label", required=True, help="Gesture label to record")
    p_collect.add_argument("--out", required=True, help="Path to output CSV")
    p_collect.add_argument("--cam", type=int, default=0, help="Camera index")

    p_train = sub.add_parser("train", help="Train a gesture classifier from CSV")
    p_train.add_argument("--data", required=True, help="Path to dataset CSV")
    p_train.add_argument("--model", required=True, help="Path to save model.pkl")

    p_infer = sub.add_parser("infer", help="Run real-time inference with a trained model")
    p_infer.add_argument("--model", required=True, help="Path to model.pkl")
    p_infer.add_argument("--cam", type=int, default=0, help="Camera index")

    args = parser.parse_args()

    if args.mode == "collect":
        run_collect(args.label, args.out, cam_index=args.cam)
    elif args.mode == "train":
        train_model(args.data, args.model)
    elif args.mode == "infer":
        rt = RealtimeInference(args.model)
        rt.run(cam_index=args.cam)

if __name__ == "__main__":
    main()
