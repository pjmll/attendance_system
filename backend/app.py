# -*- coding: utf-8 -*-
import base64
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

try:
    from active_liveness import issue_challenge, mark_challenge_used, peek_challenge, verify_action_frames
    from config import Config, LIVENESS_METHODS
    from face_detection import FaceDetector
    from liveness_detection import LivenessDetector
    from models import DatabaseManager
    from weight_manager import WeightManager
except ModuleNotFoundError:
    from backend.active_liveness import issue_challenge, mark_challenge_used, peek_challenge, verify_action_frames
    from backend.config import Config, LIVENESS_METHODS
    from backend.face_detection import FaceDetector
    from backend.liveness_detection import LivenessDetector
    from backend.models import DatabaseManager
    from backend.weight_manager import WeightManager


app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
CORS(app)

Config.init_directories()
db = DatabaseManager(Config.DATABASE_PATH)
weight_manager = WeightManager()
startup_weight_status = weight_manager.ensure_startup_weights(
    Config.DEEPFACE_MODEL_NAME,
    allow_download=Config.AUTO_DOWNLOAD_WEIGHTS_ON_STARTUP,
)
face_detector = FaceDetector(db)
liveness_detector = LivenessDetector()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def pick_available_port(preferred_port, host="127.0.0.1", attempts=20):
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"无法找到可用端口，起始端口: {preferred_port}")


def api_error(message, status=400, **extra):
    payload = {"success": False, "error": message, "timestamp": now_iso()}
    payload.update(extra)
    return jsonify(payload), status


def base64_to_image(base64_string):
    if not base64_string:
        return None
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    try:
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def image_to_base64(image):
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def decode_action_frame_list(raw_list, max_frames):
    """将前端上传的 base64 帧列表解码为 BGR 图像列表（最多 max_frames 张）。"""
    out = []
    if not isinstance(raw_list, list):
        return out
    for item in raw_list[:max_frames]:
        img = base64_to_image(item)
        if img is not None:
            out.append(img)
    return out


def sanitize_student_payload(data):
    student_id = str((data or {}).get("student_id", "")).strip()
    name = str((data or {}).get("name", "")).strip()
    major = str((data or {}).get("major", "")).strip()
    gender = str((data or {}).get("gender", "")).strip()
    if not student_id or not name:
        return None, "学号、姓名不能为空"
    if not major:
        return None, "专业不能为空"
    if gender not in ("男", "女"):
        return None, "性别必须为男或女"
    major = standardize_major(major)
    return {"student_id": student_id, "name": name, "major": major, "gender": gender}, None


def standardize_major(raw):
    """将专业名称标准化，按优先级：实验/试验 > 信安/信息安全 > 网安/网络空间安全"""
    if not raw:
        return raw
    if "试验" in raw or "实验" in raw:
        return "网络空间安全试验班"
    if "信息安全" in raw or "信安" in raw:
        return "信息安全"
    if "网安" in raw or "网络空间安全" in raw:
        return "网络空间安全"
    return raw


def parse_filename(filename):
    """从文件名解析学号-姓名-专业-性别，返回 (student_id, name, major, gender) 或 None"""
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split("-")
    if len(parts) < 4:
        return None
    student_id = parts[0].strip()
    name = parts[1].strip()
    major = standardize_major(parts[2].strip())
    gender_raw = parts[3].strip()
    if gender_raw in ("男", "male", "Male", "M", "m"):
        gender = "男"
    elif gender_raw in ("女", "female", "Female", "F", "f"):
        gender = "女"
    else:
        gender = gender_raw
    if not student_id or not name:
        return None
    return student_id, name, major, gender


def save_snapshot(image, prefix):
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(Config.SNAPSHOT_PATH, filename)
    cv2.imwrite(path, image)
    return path


def run_task_with_timeout(task_name, func, timeout_seconds, retries=0, backoff_seconds=0.0):
    last_error = None
    for attempt in range(retries + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
        try:
            result = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False, cancel_futures=True)
            return True, result, {
                "task": task_name,
                "attempts": attempt + 1,
                "timed_out": False,
            }
        except FuturesTimeoutError:
            last_error = {
                "task": task_name,
                "attempts": attempt + 1,
                "timed_out": True,
                "timeout_seconds": timeout_seconds,
            }
            future.cancel()
        except Exception as exc:
            last_error = {
                "task": task_name,
                "attempts": attempt + 1,
                "timed_out": False,
                "error": str(exc),
            }
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if attempt < retries and backoff_seconds > 0:
            time.sleep(backoff_seconds)

    return False, None, last_error or {"task": task_name, "attempts": retries + 1}


def student_row_to_dict(row):
    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "name": row["name"],
        "major": row["major"] or "",
        "gender": row["gender"] or "",
        "has_face_encoding": bool(row["face_encoding"]),
        "photo_path": row["photo_path"],
        "face_quality_score": float(row["face_quality_score"] or 0),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def record_row_to_dict(row):
    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "student_name": row["student_name"],
        "class_name": row["class_name"],
        "check_time": row["check_time"],
        "detection_method": row["detection_method"],
        "confidence": float(row["confidence"] or 0),
        "liveness_score": float(row["liveness_score"] or 0),
        "status": row["status"],
        "emotion": row["emotion"] or Config.DEFAULT_EMOTION_LABEL,
        "source": row["source"],
        "activity_name": row["activity_name"] or "",
    }


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api", methods=["GET"])
def api_home():
    return jsonify(
        {
            "success": True,
            "name": "班级考勤系统 API",
            "version": "3.1",
            "features": [
                "浏览器摄像头考勤",
                "DeepFace 人脸识别",
                "随机动作活体（多帧轨迹）",
                "本地静态活体筛查",
                "合照批量识别",
                "情绪统计与报表",
            ],
            "timestamp": now_iso(),
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "success": True,
            "status": "healthy",
            "deepface_enabled": face_detector.deepface_enabled,
            "face_database_count": len(face_detector.known_face_ids),
            "timestamp": now_iso(),
        }
    )


@app.route("/api/system_status", methods=["GET"])
def system_status():
    return jsonify(
        {
            "success": True,
            "status": "running",
            "deepface_enabled": face_detector.deepface_enabled,
            "face_database_count": len(face_detector.known_face_ids),
            "liveness_threshold": Config.PASSIVE_LIVENESS_THRESHOLD,
            "local_antispoof_available": liveness_detector.local_antispoof_available,
            "deepface_weight_dir": Config.DEEPFACE_WEIGHT_DIR,
            "startup_weight_status": startup_weight_status,
            "auto_download_weights_on_startup": Config.AUTO_DOWNLOAD_WEIGHTS_ON_STARTUP,
            "default_liveness_method": "active",
            "supported_liveness_methods": list(LIVENESS_METHODS.keys()),
            "face_recognition_timeout_seconds": Config.FACE_RECOGNITION_TIMEOUT_SECONDS,
            "liveness_timeout_seconds": Config.LIVENESS_TIMEOUT_SECONDS,
            "recognition_retry_count": Config.RECOGNITION_RETRY_COUNT,
            "passive_fallback_on_timeout": Config.ENABLE_PASSIVE_FALLBACK_ON_TIMEOUT,
            "content_safety_boundary": Config.SAFETY_BOUNDARY,
            "timestamp": now_iso(),
        }
    )


@app.route("/api/liveness_methods", methods=["GET"])
def get_liveness_methods():
    return jsonify(
        {
            "success": True,
            "methods": list(LIVENESS_METHODS.values()),
            "default_method": "active",
            "timestamp": now_iso(),
        }
    )


@app.route("/api/liveness_challenge", methods=["POST"])
def create_liveness_challenge():
    """签发随机动作挑战，供网页端连拍前拉取。"""
    cid, action, prompt = issue_challenge()
    return jsonify(
        {
            "success": True,
            "challenge_id": cid,
            "action": action,
            "prompt": prompt,
            "expires_in_seconds": int(Config.ACTIVE_CHALLENGE_TTL_SECONDS),
            "min_frames": Config.ACTIVE_BURST_MIN_FRAMES,
            "max_frames": Config.ACTIVE_BURST_MAX_FRAMES,
            "timestamp": now_iso(),
        }
    )


@app.route("/api/weights/status", methods=["GET"])
def weight_status():
    status = weight_manager.list_status()
    return jsonify({"success": True, "weights": status, "timestamp": now_iso()})


@app.route("/api/weights/download", methods=["POST"])
def download_weights():
    data = request.get_json(silent=True) or {}
    model_names = data.get("models") or [Config.DEEPFACE_MODEL_NAME]
    include_antispoof = bool(data.get("include_antispoof", True))

    downloaded = []
    for model_name in model_names:
        path, available = weight_manager.ensure_model_weight(model_name)
        downloaded.append(
            {
                "model": model_name,
                "path": path,
                "available": available,
            }
        )

    antispoof = weight_manager.ensure_antispoof_weights() if include_antispoof else []
    liveness_detector.local_antispoof_available = liveness_detector._check_local_antispoof_weights()
    return jsonify(
        {
            "success": True,
            "message": "已检查并同步项目内权重目录",
            "models": downloaded,
            "antispoof": antispoof,
            "timestamp": now_iso(),
        }
    )


@app.route("/api/safety_guidelines", methods=["GET"])
def safety_guidelines():
    return jsonify(
        {
            "success": True,
            "boundary": Config.SAFETY_BOUNDARY,
            "liveness_hint": Config.LIVENESS_HINT,
            "privacy_controls": [
                "仅保留必要的人脸特征和考勤快照",
                "建议启用 HTTPS、角色鉴权和访问审计",
                "建议对原图和报表设置存储周期并定期清理",
            ],
            "timestamp": now_iso(),
        }
    )


@app.route("/api/students", methods=["GET"])
def get_students():
    students = [student_row_to_dict(row) for row in db.get_all_students()]
    return jsonify({"success": True, "students": students, "total_count": len(students), "timestamp": now_iso()})


@app.route("/api/add_student", methods=["POST"])
def add_student():
    data = request.get_json(silent=True) or {}
    image = base64_to_image(data.get("image"))
    upload_mode = str(data.get("upload_mode", "camera")).strip() or "camera"
    payload, error = sanitize_student_payload(data)
    if error:
        return api_error(error)
    if image is None:
        return api_error("请上传或采集清晰的人脸照片")
    if db.get_student(payload["student_id"]):
        return api_error("学号已存在")

    photo_path = os.path.join(Config.FACE_DB_PATH, f'{payload["student_id"]}_{payload["name"]}.jpg')
    success, message = db.add_student(
        payload["student_id"],
        payload["name"],
        payload["major"],
        payload["gender"],
        photo_path=photo_path,
    )
    if not success:
        return api_error(message)

    face_ok, face_message = face_detector.add_face_to_database(
        image,
        payload["student_id"],
        payload["name"],
    )
    if not face_ok:
        db.delete_student(payload["student_id"])
        return api_error(face_message)

    face_detector.reload_face_database()
    return jsonify(
        {
            "success": True,
            "message": "学生录入成功",
            "student": payload,
            "upload_mode": upload_mode,
            "face_message": face_message,
            "timestamp": now_iso(),
        }
    )


@app.route("/api/delete_student/<student_id>", methods=["DELETE"])
def delete_student(student_id):
    success, message = db.delete_student(student_id)
    if not success:
        return api_error(message, 404)
    face_detector.reload_face_database()
    return jsonify({"success": True, "message": message, "student_id": student_id, "timestamp": now_iso()})


@app.route("/api/batch_upload_faces", methods=["POST"])
def batch_upload_faces():
    if "files" not in request.files:
        return api_error("请选择要上传的人脸照片文件")

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return api_error("未选择任何文件")

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    max_file_size = 12 * 1024 * 1024  # 单张照片 12MB 上限
    results = []
    success_count = 0
    fail_count = 0

    for file in files:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_ext:
            results.append({"filename": filename, "success": False, "error": f"不支持的文件格式: {ext}"})
            fail_count += 1
            continue

        # 单张照片大小校验
        file_bytes = file.read()
        if len(file_bytes) > max_file_size:
            results.append({"filename": filename, "success": False, "error": f"单张照片不能超过 12MB（当前 {round(len(file_bytes) / 1024 / 1024, 1)}MB）"})
            fail_count += 1
            continue

        parsed = parse_filename(filename)
        if parsed is None:
            results.append({"filename": filename, "success": False, "error": "文件名格式错误，应为: 学号-姓名-专业-性别.jpg"})
            fail_count += 1
            continue

        student_id, name, major, gender = parsed

        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                results.append({"filename": filename, "success": False, "error": "图片解析失败"})
                fail_count += 1
                continue
        except Exception:
            results.append({"filename": filename, "success": False, "error": "读取文件失败"})
            fail_count += 1
            continue

        photo_path = os.path.join(Config.FACE_DB_PATH, f"{student_id}_{name}.jpg")
        cv2.imwrite(photo_path, image)

        db.upsert_student(student_id, name, major, gender, photo_path=photo_path)

        face_ok, face_message = face_detector.add_face_to_database(image, student_id, name)
        if not face_ok:
            results.append({
                "filename": filename,
                "success": False,
                "student_id": student_id,
                "name": name,
                "error": face_message,
            })
            fail_count += 1
            continue

        results.append({
            "filename": filename,
            "success": True,
            "student_id": student_id,
            "name": name,
            "major": major,
            "gender": gender,
            "message": face_message,
        })
        success_count += 1

    face_detector.reload_face_database()
    return jsonify({
        "success": True,
        "total": len(results),
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
        "timestamp": now_iso(),
    })


@app.route("/api/detect_face", methods=["POST"])
def detect_face():
    data = request.get_json(silent=True) or {}
    image = base64_to_image(data.get("image"))
    if image is None:
        return api_error("图像解析失败")

    ok, faces, task_info = run_task_with_timeout(
        "detect_face",
        lambda: face_detector.recognize_face(image),
        Config.FACE_RECOGNITION_TIMEOUT_SECONDS,
        retries=Config.RECOGNITION_RETRY_COUNT,
        backoff_seconds=Config.RECOGNITION_RETRY_BACKOFF_SECONDS,
    )
    if not ok:
        return api_error(
            "人脸识别超时，请稍后重试",
            504,
            retryable=True,
            details=task_info,
        )

    annotated_image = face_detector.draw_face_boxes(image, faces) if data.get("draw_boxes") else None
    return jsonify(
        {
            "success": True,
            "face_count": len(faces),
            "faces": faces,
            "processing": {"face_recognition": task_info},
            "annotated_image": image_to_base64(annotated_image) if annotated_image is not None else None,
            "timestamp": now_iso(),
        }
    )


@app.route("/api/liveness_detection", methods=["POST"])
def liveness_detection():
    data = request.get_json(silent=True) or {}
    method = data.get("method", "passive")
    if method not in LIVENESS_METHODS:
        return api_error("不支持的活体检测方法")
    if method == "active":
        return api_error(
            "动作活体请在「实时考勤」中完成：选择「随机动作活体」后按提示连拍提交",
            400,
        )

    image = base64_to_image(data.get("image"))
    if image is None:
        return api_error("图像解析失败")
    ok, result, task_info = run_task_with_timeout(
        "liveness_detection",
        lambda: liveness_detector.comprehensive_liveness_detection(image, method),
        Config.LIVENESS_TIMEOUT_SECONDS,
        retries=Config.RECOGNITION_RETRY_COUNT,
        backoff_seconds=Config.RECOGNITION_RETRY_BACKOFF_SECONDS,
    )
    if not ok:
        fallback_result = None
        if method == "hybrid" and Config.ENABLE_PASSIVE_FALLBACK_ON_TIMEOUT:
            fallback_result = liveness_detector.passive_liveness_detection(image)
            fallback_result["message"] = "混合活体检测超时，已回退为本地静态活体筛查结果"
            fallback_result["details"] = {
                **fallback_result.get("details", {}),
                "fallback_applied": True,
                "fallback_reason": "liveness_task_timeout",
                "timed_out_task": task_info,
            }
        if fallback_result is None:
            return api_error(
                "活体检测超时，请稍后重试",
                504,
                retryable=True,
                details=task_info,
            )
        return jsonify({
            "success": True,
            **fallback_result,
            "method": method,
            "processing": {"liveness": task_info},
            "timestamp": now_iso(),
        })
    return jsonify({"success": True, **result, "method": method, "processing": {"liveness": task_info}, "timestamp": now_iso()})


@app.route("/api/attendance", methods=["POST"])
def attendance():
    data = request.get_json(silent=True) or {}
    method = data.get("liveness_method", "passive")
    if method not in LIVENESS_METHODS:
        return api_error("不支持的活体检测方法")

    max_f = int(getattr(Config, "ACTIVE_BURST_MAX_FRAMES", 28))
    image = None
    motion_details = None
    motion_msg = None

    if method == "active":
        challenge_id = str(data.get("challenge_id", "")).strip()
        if not challenge_id:
            return api_error("缺少 challenge_id，请重新点击考勤以获取动作指令")
        rec, ch_err = peek_challenge(challenge_id)
        if ch_err in ("missing", "expired"):
            return api_error("动作挑战无效或已过期，请重新点击考勤获取新指令")
        if ch_err == "used":
            return api_error("该挑战已使用，请重新点击考勤获取新指令")
        action_expected = rec["action"]
        bgr_list = decode_action_frame_list(data.get("action_frames"), max_f)
        min_need = int(getattr(Config, "ACTIVE_BURST_MIN_FRAMES", 8))
        if len(bgr_list) < min_need:
            return api_error(
                f"有效动作帧过少（{len(bgr_list)}/{min_need}），请提高环境亮度或靠近摄像头后重试"
            )
        motion_ok, motion_msg, motion_details = verify_action_frames(action_expected, bgr_list)
        if not motion_ok:
            return api_error(
                motion_msg or "动作活体未通过",
                details={"active_motion": motion_details},
            )
        if not mark_challenge_used(challenge_id):
            return api_error("挑战状态异常，请重试")
        image = bgr_list[-1]
    else:
        image = base64_to_image(data.get("image"))
        if image is None:
            return api_error("图像解析失败，请重新拍摄")

    face_ok, face_results, face_task = run_task_with_timeout(
        "attendance_face_recognition",
        lambda: face_detector.recognize_face(image),
        Config.FACE_RECOGNITION_TIMEOUT_SECONDS,
        retries=Config.RECOGNITION_RETRY_COUNT,
        backoff_seconds=Config.RECOGNITION_RETRY_BACKOFF_SECONDS,
    )
    if not face_ok:
        return api_error(
            "考勤识别超时，请重新拍摄后重试",
            504,
            retryable=True,
            details={"face_recognition": face_task},
        )

    if not face_results:
        return api_error("未检测到可识别的人脸")
    if len(face_results) > 1:
        return api_error("基础考勤只允许单人入镜，请使用合照识别处理多人场景")

    passive_method = "passive"
    liveness_ok, liveness_result, liveness_task = run_task_with_timeout(
        "attendance_liveness_detection",
        lambda: liveness_detector.comprehensive_liveness_detection(image, passive_method),
        Config.LIVENESS_TIMEOUT_SECONDS,
        retries=Config.RECOGNITION_RETRY_COUNT,
        backoff_seconds=Config.RECOGNITION_RETRY_BACKOFF_SECONDS,
    )
    if not liveness_ok:
        if method == "hybrid" and Config.ENABLE_PASSIVE_FALLBACK_ON_TIMEOUT:
            liveness_result = liveness_detector.passive_liveness_detection(image)
            liveness_result["message"] = "活体检测超时，已回退为本地静态活体筛查结果"
            liveness_result["details"] = {
                **liveness_result.get("details", {}),
                "fallback_applied": True,
                "fallback_reason": "attendance_liveness_timeout",
                "timed_out_task": liveness_task,
            }
        else:
            return api_error(
                "活体检测超时，请稍后重试",
                504,
                retryable=True,
                details={"liveness": liveness_task},
            )

    if method == "active" and motion_details is not None:
        merged_details = dict(liveness_result.get("details") or {})
        merged_details["active_motion"] = motion_details
        merged_details["active_motion_message"] = motion_msg
        boosted = min(1.0, float(liveness_result["final_score"]) + 0.08)
        liveness_result = {
            **liveness_result,
            "details": merged_details,
            "final_score": round(boosted, 4),
            "message": (liveness_result.get("message") or "") + "；随机动作轨迹已通过",
        }

    if not liveness_result["is_live"]:
        return api_error(
            "活体检测未通过",
            liveness_score=liveness_result["final_score"],
            details=liveness_result["details"],
            boundary=liveness_result["boundary"],
            # message=liveness_result["message"],
        )

    result = face_results[0]
    if not result["matched"]:
        return api_error("未匹配到学生档案，请先录入该学生的人脸信息")

    snapshot_path = save_snapshot(image, f"attendance_{result['student_id']}")
    db.add_attendance_record(
        student_id=result["student_id"],
        student_name=result["student_name"],
        class_name=result["class_name"],
        detection_method=method,
        confidence=result["confidence"],
        liveness_score=liveness_result["final_score"],
        emotion=result["emotion"],
        snapshot_path=snapshot_path,
        source="attendance",
    )

    return jsonify(
        {
            "success": True,
            "attendance_results": [result],
            "liveness_score": liveness_result["final_score"],
            "liveness_method": method,
            "liveness_method_name": LIVENESS_METHODS[method]["name"],
            "emotion": result["emotion"],
            "message": "考勤成功",
            "boundary": Config.LIVENESS_HINT,
            "processing": {
                "face_recognition": face_task,
                "liveness": liveness_task,
            },
            "timestamp": now_iso(),
        }
    )


@app.route("/api/group_recognition", methods=["POST"])
def group_recognition():
    data = request.get_json(silent=True) or {}
    image = base64_to_image(data.get("image"))
    if image is None:
        return api_error("请上传合照图片")

    # 动态缩放合照：限制最大长边，极大提升处理速度
    max_side = 1280
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    activity_name = str(data.get("activity_name", "")).strip() or "课堂活动"
    major_filter = str(data.get("major", "")).strip()
    gender_filter = str(data.get("gender", "")).strip()
    face_ok, face_results, face_task = run_task_with_timeout(
        "group_face_recognition",
        lambda: face_detector.recognize_face(image),
        Config.FACE_RECOGNITION_TIMEOUT_SECONDS,
        retries=Config.RECOGNITION_RETRY_COUNT,
        backoff_seconds=Config.RECOGNITION_RETRY_BACKOFF_SECONDS,
    )
    if not face_ok:
        return api_error(
            "合照识别超时，请压缩图片或稍后重试",
            504,
            retryable=True,
            details={"face_recognition": face_task},
        )

    if not face_results:
        return api_error("未在合照中检测到人脸")

    # 构建专业筛选关键词列表
    major_keywords = []
    if major_filter == "网络空间安全":
        major_keywords = ["网络空间安全"]
    elif major_filter == "信息安全":
        major_keywords = ["信息安全"]

    # 按专业和性别筛选匹配结果
    def matches_filter(item):
        if not item["matched"]:
            return False
        if major_keywords:
            student_major = item.get("class_name", "")
            if not any(kw in student_major for kw in major_keywords):
                return False
        if gender_filter and gender_filter in ("男", "女"):
            if item.get("gender", "") != gender_filter:
                return False
        return True

    unique_matches = {}
    for item in face_results:
        if matches_filter(item):
            unique_matches[item["student_id"]] = item

    # 构建筛选描述
    filter_parts = []
    if major_filter and major_filter != "不限":
        filter_parts.append(f"专业: {major_filter}")
    if gender_filter and gender_filter != "不限":
        filter_parts.append(f"性别: {gender_filter}")
    filter_desc = ", ".join(filter_parts) if filter_parts else "不限"

    snapshot_path = save_snapshot(image, "group")
    session_id = db.add_group_session(
        activity_name=activity_name,
        class_name=filter_desc,
        total_faces=len(face_results),
        matched_faces=len(unique_matches),
        image_path=snapshot_path,
        results=[item for item in face_results if matches_filter(item) or not item["matched"]],
    )

    for item in unique_matches.values():
        db.add_attendance_record(
            student_id=item["student_id"],
            student_name=item["student_name"],
            class_name=item["class_name"],
            detection_method="group_recognition",
            confidence=item["confidence"],
            liveness_score=0.0,
            emotion=item["emotion"],
            snapshot_path=snapshot_path,
            source="group",
            activity_name=activity_name,
        )

    filtered_results = [item for item in face_results if matches_filter(item) or not item["matched"]]
    annotated = face_detector.draw_face_boxes(image, filtered_results)
    emotion_counter = {}
    for item in filtered_results:
        emotion_counter[item["emotion"]] = emotion_counter.get(item["emotion"], 0) + 1

    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "activity_name": activity_name,
            "major_filter": major_filter or "不限",
            "gender_filter": gender_filter or "不限",
            "total_faces": len(face_results),
            "matched_count": len(unique_matches),
            "unknown_count": len(face_results) - len(unique_matches),
            "results": filtered_results,
            "emotion_summary": [{"name": key, "value": value} for key, value in emotion_counter.items()],
            "annotated_image": image_to_base64(annotated),
            "processing": {"face_recognition": face_task},
            "boundary": "合照识别适用于名单生成与课堂统计，不建议用于个体惩戒和持续监控。",
            "timestamp": now_iso(),
        }
    )


@app.route("/api/attendance_records", methods=["GET"])
def attendance_records():
    date = request.args.get("date")
    rows = db.get_attendance_by_date(date) if date else db.get_today_attendance()
    records = [record_row_to_dict(row) for row in rows]
    return jsonify(
        {
            "success": True,
            "records": records,
            "total_count": len(records),
            "date": date,
            "timestamp": now_iso(),
        }
    )


@app.route("/api/attendance_records", methods=["DELETE"])
def delete_attendance_records():
    date = request.args.get("date")
    success, message = db.delete_attendance_records(date)
    return jsonify({"success": success, "message": message, "timestamp": now_iso()})


@app.route("/api/attendance_summary", methods=["GET"])
def attendance_summary():
    date = request.args.get("date")
    summary = db.get_attendance_summary(date)
    return jsonify({"success": True, "summary": summary, "timestamp": now_iso()})


@app.route("/api/emotion_stats", methods=["GET"])
def emotion_stats():
    date = request.args.get("date")
    data = db.get_emotion_stats(date)
    return jsonify({"success": True, "data": data, "timestamp": now_iso()})


@app.route("/api/group_reports", methods=["GET"])
def group_reports():
    return jsonify({"success": True, "reports": db.get_group_reports(), "timestamp": now_iso()})


@app.route("/api/activity_stats", methods=["GET"])
def activity_stats():
    date = request.args.get("date")
    limit = request.args.get("limit", default=12, type=int) or 12
    limit = max(1, min(limit, 50))
    data = db.get_activity_stats(date=date, limit=limit)
    return jsonify({"success": True, **data, "date": date, "timestamp": now_iso()})


@app.route("/api/export_attendance", methods=["GET"])
def export_attendance():
    date = request.args.get("date")
    csv_content = db.export_attendance_to_csv(date)
    filename = f"attendance_{date or datetime.now().strftime('%Y-%m-%d')}.csv"
    return Response(
        csv_content,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@app.errorhandler(404)
def not_found(_error):
    return api_error("接口不存在", 404)


@app.errorhandler(413)
def too_large(_error):
    return api_error("单张照片大小不能超过 12MB", 413)


@app.errorhandler(500)
def internal_error(error):
    return api_error(f"服务内部异常: {error}", 500)


if __name__ == "__main__":
    preferred_port = int(os.environ.get("PORT", 5001))
    selected_port = pick_available_port(preferred_port, host="0.0.0.0")
    if selected_port != preferred_port:
        print(f"Port {preferred_port} is in use, fallback to {selected_port}")
    print(f"Classroom attendance system is running at http://127.0.0.1:{selected_port}")
    app.run(host="0.0.0.0", port=selected_port, debug=False)
