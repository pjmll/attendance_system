import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

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
except Exception as exc:  # pragma: no cover
    DeepFace = None
    print(f"DeepFace import failed: {exc}")

import insightface
from insightface.app import FaceAnalysis

EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}


def detect_mock_emotion(image):
    labels = [
        "neutral",
        "focused",
        "happy",
        "neutral",
        "surprise",
    ]
    try:
        idx = int(np.asarray(image).sum()) % len(labels)
        return labels[idx]
    except Exception:
        return Config.DEFAULT_EMOTION_LABEL


class FaceDetector:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_majors = []
        self.known_face_genders = []
        self.deepface_enabled = DeepFace is not None
        self.df_model_name = Config.DEEPFACE_MODEL_NAME
        self.df_detector_backend = Config.DEEPFACE_DETECTOR_BACKEND
        self.df_distance_metric = Config.DEEPFACE_DISTANCE_METRIC
        self.df_max_distance = Config.DEEPFACE_MAX_DISTANCE
        self.weight_manager = WeightManager()
        self.embedding_dim = Config.DEEPFACE_MODEL_DIMS.get(self.df_model_name)
        self.weight_manager.ensure_model_weight(
            self.df_model_name,
            allow_download=Config.AUTO_DOWNLOAD_WEIGHTS_ON_STARTUP,
        )
        
        print("Initializing InsightFace model...")
        self.insight_app = FaceAnalysis(name='buffalo_l', root=os.path.join(Config.DEEPFACE_HOME, '.insightface'))
        self.insight_app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.load_face_database()

    @staticmethod
    def image_to_base64(image):
        if isinstance(image, np.ndarray):
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                return None
            return base64.b64encode(buffer).decode("utf-8")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def load_face_database(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_majors = []
        self.known_face_genders = []
        if not self.db_manager:
            return

        for item in self.db_manager.get_all_face_encodings():
            encoding = item["encoding"]
            if encoding is None:
                continue
            stored_model = item.get("face_model")
            stored_dim = int(item.get("face_embedding_dim") or len(encoding))
            if stored_model == self.df_model_name and stored_dim == len(encoding) == self.embedding_dim:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(item["name"])
                self.known_face_ids.append(item["student_id"])
                self.known_face_majors.append(item["major"])
                self.known_face_genders.append(item.get("gender", ""))
                continue

            refreshed = self._rebuild_student_encoding(item)
            if refreshed is not None:
                self.known_face_encodings.append(refreshed)
                self.known_face_names.append(item["name"])
                self.known_face_ids.append(item["student_id"])
                self.known_face_majors.append(item["major"])
                self.known_face_genders.append(item.get("gender", ""))

    def reload_face_database(self):
        self.load_face_database()

    def _extract_faces(self, image_bgr):
        try:
            faces = self.insight_app.get(image_bgr)
        except Exception as exc:
            print(f"InsightFace extract_faces failed: {exc}")
            return []

        valid_faces = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            if w < Config.MIN_FACE_SIZE:
                continue
                
            # Expand bounding box slightly for DeepFace emotion crop later
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            
            face_data = {
                "facial_area": {"x": x, "y": y, "w": w, "h": h},
                "face": image_bgr[max(0, y-pad_h):y2+pad_h, max(0, x-pad_w):x2+pad_w], # approximate crop for emotion
                "embedding": face.normed_embedding if face.normed_embedding is not None else face.embedding,
                "gender": face.sex, 
            }
            valid_faces.append(face_data)
        return valid_faces

    def _embedding_from_face_rgb(self, face_rgb):
        # We no longer extract embedding here since InsightFace returns it directly during _extract_faces.
        # This method is obsolete but kept for signature compatibility if it's called anywhere.
        return None

    def _rebuild_student_encoding(self, item):
        photo_path = item.get("photo_path")
        if not photo_path:
            return None
        if not os.path.isabs(photo_path):
            candidate = os.path.join(Config.FACE_DB_PATH, os.path.basename(photo_path))
        else:
            candidate = photo_path
        if not os.path.exists(candidate):
            candidate = os.path.join(Config.FACE_DB_PATH, f"{item['student_id']}_{item['name']}.jpg")
        if not os.path.exists(candidate):
            print(f"Skip rebuilding encoding for {item['student_id']}: photo missing")
            return None

        image = cv2.imread(candidate)
        if image is None:
            return None

        refreshed = self._get_single_embedding(image)
        if refreshed is None:
            print(f"Skip rebuilding encoding for {item['student_id']}: no single face detected")
            return None

        self.db_manager.update_student_face_encoding(
            item["student_id"],
            refreshed,
            face_quality_score=self.get_face_quality_score(image),
            face_model=self.df_model_name,
        )
        return refreshed

    def _get_single_embedding(self, image_bgr):
        faces = self._extract_faces(image_bgr)
        if len(faces) != 1:
            return None
        return faces[0]["embedding"]

    @staticmethod
    def _cosine_distance(vec_a, vec_b):
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(1.0 - np.dot(a, b) / denom)

    def _distance(self, vec_a, vec_b):
        return self._cosine_distance(vec_a, vec_b)

    @staticmethod
    def _clip_area(area, image_shape, padding_ratio=0.18):
        h, w = image_shape[:2]
        x = int(area.get("x", 0) or 0)
        y = int(area.get("y", 0) or 0)
        fw = int(area.get("w", 0) or 0)
        fh = int(area.get("h", 0) or 0)
        pad_x = int(fw * padding_ratio)
        pad_y = int(fh * padding_ratio)
        left = max(x - pad_x, 0)
        top = max(y - pad_y, 0)
        right = min(x + fw + pad_x, w)
        bottom = min(y + fh + pad_y, h)
        return left, top, right, bottom

    def _crop_face_for_emotion(self, image_bgr, area):
        left, top, right, bottom = self._clip_area(area, image_bgr.shape)
        crop = image_bgr[top:bottom, left:right]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        return crop

    def _dominant_emotion_zh(self, image_bgr, area):
        if not self.deepface_enabled:
            return detect_mock_emotion(image_bgr)

        face_crop = self._crop_face_for_emotion(image_bgr, area)
        if face_crop is None:
            return Config.DEFAULT_EMOTION_LABEL

        try:
            analysis = DeepFace.analyze(
                img_path=face_crop,
                actions=["emotion"],
                detector_backend=self.df_detector_backend,
                enforce_detection=False,
                align=True,
                silent=True,
            )
            if isinstance(analysis, list):
                analysis = analysis[0] if analysis else {}
            emotion_scores = analysis.get("emotion") if isinstance(analysis.get("emotion"), dict) else {}
            if emotion_scores:
                best_label = max(emotion_scores, key=emotion_scores.get)
                best_score = float(emotion_scores[best_label])
                sorted_scores = sorted(emotion_scores.values(), reverse=True)
                margin = best_score - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
                if best_score < 35 or margin < 8:
                    return Config.DEFAULT_EMOTION_LABEL
                return EMOTION_MAP.get(str(best_label).lower(), str(best_label))

            dominant = analysis.get("dominant_emotion")
            if dominant:
                return EMOTION_MAP.get(str(dominant).lower(), str(dominant))
        except Exception as exc:
            print(f"DeepFace analyze failed: {exc}")

        return detect_mock_emotion(face_crop)

    def get_face_quality_score(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
            faces = self._extract_faces(image)
            if not faces:
                return 0.0
            area = faces[0]["facial_area"]
            face_size = (int(area.get("w", 0)) * int(area.get("h", 0))) / float(image.shape[0] * image.shape[1] + 1)
            size_score = min(face_size * 6.0, 1.0)
            return round(sharpness * 0.7 + size_score * 0.3, 4)
        except Exception:
            return 0.0

    def add_face_to_database(self, image, student_id, student_name):
        face_encoding = self._get_single_embedding(image)
        if face_encoding is None:
            return False, "请确保画面中只有一张清晰正脸"

        filename = f"{student_id}_{student_name}.jpg"
        image_path = os.path.join(Config.FACE_DB_PATH, filename)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image.save(image_path)

        quality_score = self.get_face_quality_score(image)
        if self.db_manager:
            updated = self.db_manager.update_student_face_encoding(
                student_id,
                face_encoding,
                quality_score,
                face_model=self.df_model_name,
            )
            if not updated:
                return False, "学生信息已创建，但写入人脸特征失败"

        if student_id in self.known_face_ids:
            idx = self.known_face_ids.index(student_id)
            self.known_face_encodings[idx] = face_encoding
            self.known_face_names[idx] = student_name
        else:
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(student_name)
            self.known_face_ids.append(student_id)
            student = self.db_manager.get_student(student_id) if self.db_manager else None
            self.known_face_majors.append(student["major"] if student else "")

        return True, f"已录入人脸特征，质量分数 {quality_score:.2f}"

    def recognize_face(self, image):
        faces = self._extract_faces(image)
        if not faces:
            return []

        # Phase 1: face matching (fast, sequential)
        results = []
        for face in faces[: Config.MAX_GROUP_FACES]:
            area = face.get("facial_area") or {}
            x = int(area.get("x", 0) or 0)
            y = int(area.get("y", 0) or 0)
            w = int(area.get("w", 0) or 0)
            h = int(area.get("h", 0) or 0)
            embedding = face.get("embedding")

            student_id = "Unknown"
            student_name = "未匹配"
            class_name = ""
            gender = ""
            confidence = 0.0
            matched = False

            if embedding is not None and self.known_face_encodings:
                distances = [self._distance(item, embedding) for item in self.known_face_encodings]
                best_index = int(np.argmin(distances))
                best_distance = float(distances[best_index])
                confidence = max(0.0, min(1.0, 1.0 - best_distance / max(self.df_max_distance, 1e-6)))
                if best_distance <= self.df_max_distance and confidence >= Config.MIN_RECOGNITION_CONFIDENCE:
                    student_id = self.known_face_ids[best_index]
                    student_name = self.known_face_names[best_index]
                    class_name = self.known_face_majors[best_index]
                    gender = self.known_face_genders[best_index]
                    matched = True

            results.append(
                {
                    "student_id": student_id,
                    "student_name": student_name,
                    "class_name": class_name,
                    "gender": gender,
                    "confidence": round(confidence, 4),
                    "emotion": None,
                    "matched": matched,
                    "face_location": {
                        "top": y,
                        "right": x + w,
                        "bottom": y + h,
                        "left": x,
                        "width": w,
                        "height": h,
                    },
                    "_area": area,
                }
            )

        # Phase 2: emotion analysis (slow, parallelized with ThreadPoolExecutor)
        if results:
            max_workers = min(len(results), 6)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._dominant_emotion_zh, image, r["_area"]): i
                    for i, r in enumerate(results)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx]["emotion"] = future.result()
                    except Exception as exc:
                        print(f"Emotion analysis failed for face {idx}: {exc}")
                        results[idx]["emotion"] = Config.DEFAULT_EMOTION_LABEL

        for r in results:
            if not r["emotion"]:
                r["emotion"] = Config.DEFAULT_EMOTION_LABEL
            r.pop("_area", None)

        return results

    def draw_face_boxes(self, image, face_results):
        output = image.copy()
        for result in face_results:
            location = result["face_location"]
            top = location["top"]
            left = location["left"]
            right = location["right"]
            bottom = location["bottom"]
            color = (42, 169, 82) if result["matched"] else (36, 99, 235)
            cv2.rectangle(output, (left, top), (right, bottom), color, 2)
            label = f"{result['student_name']} {int(result['confidence'] * 100)}%"
            cv2.rectangle(output, (left, max(top - 28, 0)), (right, top), color, -1)
            cv2.putText(
                output,
                label,
                (left + 6, max(top - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return output
