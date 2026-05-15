# -*- coding: utf-8 -*-
"""随机动作活体：签发挑战 + 多帧人脸框轨迹校验（仅依赖 OpenCV Haar）。"""
import random
import secrets
import time
from threading import Lock

import cv2
import numpy as np

try:
    from config import Config
except ModuleNotFoundError:
    from backend.config import Config

_ACTION_POOL = ("nod", "shake", "turn_left", "turn_right")

_ACTION_PROMPTS = {
    "nod": "请缓慢点头一次：下巴靠近胸口再抬起，约 3 秒内完成",
    "shake": "请缓慢左右摇头（像说「不」一样），约 3 秒内完成",
    "turn_left": "请向自己的左侧转头，再转回正视镜头，约 3 秒内完成",
    "turn_right": "请向自己的右侧转头，再转回正视镜头，约 3 秒内完成",
}

_store: dict = {}
_lock = Lock()


def action_prompt(action: str) -> str:
    return _ACTION_PROMPTS.get(action, "请按界面提示完成动作")


def _prune_stale():
    ttl = float(getattr(Config, "ACTIVE_CHALLENGE_TTL_SECONDS", 120))
    now = time.time()
    dead = [k for k, v in _store.items() if now - v.get("t", 0) > ttl]
    for k in dead:
        _store.pop(k, None)


def issue_challenge():
    """返回 (challenge_id, action, prompt_zh)。"""
    _prune_stale()
    action = random.choice(_ACTION_POOL)
    cid = secrets.token_urlsafe(18)
    with _lock:
        _store[cid] = {"action": action, "t": time.time(), "used": False}
    return cid, action, action_prompt(action)


def peek_challenge(challenge_id: str):
    """
    返回 (record 或 None, error_code)。
    error_code: None 表示可用；否则为 'missing' | 'used' | 'expired'。
    """
    with _lock:
        _prune_stale()
        rec = _store.get(challenge_id)
        if rec is None:
            return None, "missing"
        if rec.get("used"):
            return None, "used"
        ttl = float(getattr(Config, "ACTIVE_CHALLENGE_TTL_SECONDS", 120))
        if time.time() - rec.get("t", 0) > ttl:
            _store.pop(challenge_id, None)
            return None, "expired"
        return dict(rec), None


def mark_challenge_used(challenge_id: str) -> bool:
    with _lock:
        rec = _store.get(challenge_id)
        if not rec or rec.get("used"):
            return False
        rec["used"] = True
        return True


def _largest_face_center(gray, cascade, w, h):
    min_side = max(48, int(getattr(Config, "ACTIVE_MIN_FACE_SIDE", 55)))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=4,
        minSize=(min_side, min_side),
    )
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    cx = (x + fw / 2.0) / float(w)
    cy = (y + fh / 2.0) / float(h)
    area = (fw * fh) / float(w * h)
    return cx, cy, area


def verify_action_frames(action: str, bgr_frames: list) -> tuple:
    if action not in _ACTION_PROMPTS:
        return False, "未知的动作类型", {}

    nmin = int(getattr(Config, "ACTIVE_BURST_MIN_FRAMES", 8))
    nmax = int(getattr(Config, "ACTIVE_BURST_MAX_FRAMES", 28))
    if len(bgr_frames) < nmin:
        return False, f"动作帧过少（至少需要 {nmin} 帧）", {"frames": len(bgr_frames)}
    if len(bgr_frames) > nmax:
        return False, f"动作帧过多（最多 {nmax} 帧）", {"frames": len(bgr_frames)}

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    centers = []
    for img in bgr_frames:
        if img is None or img.size == 0:
            continue
        h, w = img.shape[:2]
        if h < 64 or w < 64:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = _largest_face_center(gray, cascade, w, h)
        if r:
            centers.append((float(r[0]), float(r[1]), float(r[2])))

    min_detected = int(getattr(Config, "ACTIVE_MIN_DETECTED_FRAMES", 6))
    min_ratio = float(getattr(Config, "ACTIVE_MIN_FACE_FRAME_RATIO", 0.42))
    if len(centers) < min_detected:
        return (
            False,
            f"有效人脸帧过少（{len(centers)}/{len(bgr_frames)}），请正对镜头、提高亮度后重试",
            {"detected_frames": len(centers), "total_frames": len(bgr_frames)},
        )
    if len(centers) / float(len(bgr_frames)) < min_ratio:
        return False, "人脸在画面中不稳定，请全程保持正脸在取景框内", {"face_frame_ratio": round(len(centers) / len(bgr_frames), 3)}

    cx = np.array([c[0] for c in centers], dtype=np.float64)
    cy = np.array([c[1] for c in centers], dtype=np.float64)
    n = len(cx)

    details = {
        "detected_frames": n,
        "total_frames": len(bgr_frames),
        "cx_range": round(float(cx.max() - cx.min()), 4),
        "cy_range": round(float(cy.max() - cy.min()), 4),
    }

    nod_th = float(getattr(Config, "ACTIVE_NOD_CY_RANGE", 0.052))
    shake_th = float(getattr(Config, "ACTIVE_SHAKE_CX_RANGE", 0.068))
    turn_delta = float(getattr(Config, "ACTIVE_TURN_HEAD_DELTA", 0.038))
    turn_margin = float(getattr(Config, "ACTIVE_TURN_PEAK_EDGE_MARGIN", 0.12))

    def peak_not_edge(arg_extreme_idx, _unused_max_flag):
        lo = int(n * turn_margin)
        hi_excl = max(lo + 1, int(n * (1.0 - turn_margin)))
        pi = int(arg_extreme_idx)
        if n < 5:
            return True
        return lo <= pi < hi_excl

    if action == "nod":
        span = float(cy.max() - cy.min())
        ok = span >= nod_th
        details["nod_span"] = round(span, 4)
        return (
            ok,
            "已通过点头校验" if ok else f"点头幅度不足（垂直位移约 {round(span, 3)}，需更明显）",
            details,
        )

    if action == "shake":
        span = float(cx.max() - cx.min())
        ok = span >= shake_th
        details["shake_span"] = round(span, 4)
        return (
            ok,
            "已通过摇头校验" if ok else f"左右摆动幅度不足（水平位移约 {round(span, 3)}）",
            details,
        )

    ref_cx = float(cx[0])
    if action == "turn_left":
        # 用户向左转头：人脸在画面中整体向右移，cx 增大
        peak_i = int(np.argmax(cx))
        span = float(np.max(cx) - ref_cx)
        ok = span >= turn_delta and peak_not_edge(peak_i, True)
        details["turn_left_span"] = round(span, 4)
        details["peak_index"] = peak_i
        return (
            ok,
            "已通过向左转头校验" if ok else "未观察到足够的向左转头，请幅度稍大并再转回正视",
            details,
        )

    if action == "turn_right":
        trough_i = int(np.argmin(cx))
        span = float(ref_cx - np.min(cx))
        ok = span >= turn_delta and peak_not_edge(trough_i, False)
        details["turn_right_span"] = round(span, 4)
        details["trough_index"] = trough_i
        return (
            ok,
            "已通过向右转头校验" if ok else "未观察到足够的向右转头，请幅度稍大并再转回正视",
            details,
        )

    return False, "未知的动作类型", details
