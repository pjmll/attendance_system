import os
import time

import cv2
import numpy as np

try:
    from config import Config
except ModuleNotFoundError:
    from backend.config import Config
try:
    from weight_manager import WeightManager
except ModuleNotFoundError:
    from backend.weight_manager import WeightManager

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None


class LivenessDetector:
    def __init__(self):
        self.weight_manager = WeightManager()
        self.weight_manager.ensure_antispoof_weights(
            allow_download=Config.AUTO_DOWNLOAD_WEIGHTS_ON_STARTUP,
        )
        self.local_antispoof_available = self._check_local_antispoof_weights()

    @staticmethod
    def _check_local_antispoof_weights():
        if DeepFace is None:
            return False
        return all(
            os.path.exists(os.path.join(Config.DEEPFACE_WEIGHT_DIR, filename))
            for filename in Config.ANTISPOOF_REQUIRED_FILES
        )

    @staticmethod
    def _face_roi(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return image
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        pad_x = int(w * 0.12)
        pad_y = int(h * 0.12)
        left = max(x - pad_x, 0)
        top = max(y - pad_y, 0)
        right = min(x + w + pad_x, image.shape[1])
        bottom = min(y + h + pad_y, image.shape[0])
        return image[top:bottom, left:right]

    @staticmethod
    def _normalized_entropy(gray_face):
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256]).ravel()
        hist_sum = hist.sum()
        if hist_sum <= 0:
            return 0.0
        prob = hist / hist_sum
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        return float(min(entropy / 8.0, 1.0))

    def passive_liveness_detection(self, image):
        face = self._face_roi(image)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = float(min(laplacian_var / 220.0, 1.0))

        entropy_score = self._normalized_entropy(gray)

        brightness = float(np.mean(gray) / 255.0)
        brightness_score = max(0.0, 1.0 - abs(brightness - 0.52) / 0.42)

        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        saturation_score = min(saturation / 0.35, 1.0)

        highlights = float(np.mean((hsv[:, :, 2] > 245).astype(np.float32)))
        highlight_penalty = min(highlights * 3.2, 0.35)

        edges = cv2.Canny(gray, 70, 170)
        edge_density = float(np.mean(edges > 0))
        edge_score = min(edge_density / 0.16, 1.0)

        channels = cv2.split(face.astype(np.float32))
        channel_gap = float(
            np.mean(np.abs(channels[0] - channels[1])) +
            np.mean(np.abs(channels[1] - channels[2]))
        ) / 255.0
        recapture_penalty = max(0.0, 0.22 - channel_gap) * 1.35

        heuristic_score = (
            sharpness_score * 0.22 +
            entropy_score * 0.28 +
            brightness_score * 0.12 +
            saturation_score * 0.12 +
            edge_score * 0.26
        ) - highlight_penalty - recapture_penalty

        heuristic_score = round(float(max(0.0, min(heuristic_score, 1.0))), 4)
        antispoof_score = None
        antispoof_real = None
        anti_spoof_message = "本地反欺骗权重未就绪，当前仅使用启发式静态筛查"

        if self.local_antispoof_available:
            try:
                faces = DeepFace.extract_faces(
                    img_path=image,
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                    anti_spoofing=True,
                )
                if faces:
                    antispoof_real = bool(faces[0].get("is_real", False))
                    antispoof_score = float(faces[0].get("antispoof_score", 0.0))
                    anti_spoof_message = "已启用 DeepFace FasNet 反欺骗评分"
            except Exception as exc:
                anti_spoof_message = f"DeepFace 反欺骗调用失败: {exc}"

        if antispoof_score is not None:
            final_score = round(heuristic_score * 0.35 + antispoof_score * 0.65, 4)
            is_live = antispoof_real and final_score >= Config.PASSIVE_LIVENESS_THRESHOLD
        else:
            final_score = heuristic_score
            is_live = final_score >= Config.PASSIVE_LIVENESS_THRESHOLD

        return {
            "final_score": final_score,
            "is_live": is_live,
            "details": {
                "heuristic_score": heuristic_score,
                "sharpness": round(sharpness_score, 4),
                "entropy": round(entropy_score, 4),
                "brightness": round(brightness_score, 4),
                "saturation": round(saturation_score, 4),
                "edge_density": round(edge_score, 4),
                "highlight_penalty": round(highlight_penalty, 4),
                "recapture_penalty": round(recapture_penalty, 4),
                "antispoof_score": round(antispoof_score, 4) if antispoof_score is not None else None,
                "antispoof_real": antispoof_real,
                "antispoof_available": self.local_antispoof_available,
                "antispoof_message": anti_spoof_message,
                "threshold": Config.PASSIVE_LIVENESS_THRESHOLD,
            },
            "message": "本地静态活体筛查完成",
            "boundary": Config.LIVENESS_HINT,
        }

    def comprehensive_liveness_detection(self, image, method="passive"):
        return self.passive_liveness_detection(image)
