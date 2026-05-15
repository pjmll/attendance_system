import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEEPFACE_HOME = PROJECT_ROOT
os.environ["DEEPFACE_HOME"] = DEEPFACE_HOME


class Config:
    DATABASE_PATH = os.path.join(PROJECT_ROOT, "database", "attendance.db")
    FACE_DB_PATH = os.path.join(BASE_DIR, "static", "face_db")
    SNAPSHOT_PATH = os.path.join(BASE_DIR, "static", "snapshots")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 不限制总大小，单文件12MB在前端+后端分别校验
    REQUEST_TIMEOUT_SECONDS = 30
    AUTO_DOWNLOAD_WEIGHTS_ON_STARTUP = os.getenv("AUTO_DOWNLOAD_WEIGHTS_ON_STARTUP", "false").lower() == "true"
    FACE_RECOGNITION_TIMEOUT_SECONDS = float(os.getenv("FACE_RECOGNITION_TIMEOUT_SECONDS", "60"))
    LIVENESS_TIMEOUT_SECONDS = float(os.getenv("LIVENESS_TIMEOUT_SECONDS", "15"))
    RECOGNITION_RETRY_COUNT = max(int(os.getenv("RECOGNITION_RETRY_COUNT", "1")), 0)
    RECOGNITION_RETRY_BACKOFF_SECONDS = max(float(os.getenv("RECOGNITION_RETRY_BACKOFF_SECONDS", "0.3")), 0.0)
    ENABLE_PASSIVE_FALLBACK_ON_TIMEOUT = os.getenv("ENABLE_PASSIVE_FALLBACK_ON_TIMEOUT", "true").lower() == "true"

    MIN_FACE_SIZE = 80
    MAX_GROUP_FACES = 50

    DEEPFACE_MODEL_NAME = "InsightFace_buffalo_l"
    DEEPFACE_DETECTOR_BACKEND = "opencv"
    DEEPFACE_DISTANCE_METRIC = "cosine"
    DEEPFACE_MAX_DISTANCE = 0.6  # InsightFace embedding norm threshold is different usually, but cosine is fine
    MIN_RECOGNITION_CONFIDENCE = 0.45

    PASSIVE_LIVENESS_THRESHOLD = 0.35 #本地静态活体筛查阈值，经验值，实际部署时可根据需求调整
    DEFAULT_EMOTION_LABEL = "neutral"

    # 随机动作活体（多帧 OpenCV 轨迹 + 单帧被动筛查）
    ACTIVE_CHALLENGE_TTL_SECONDS = 120
    ACTIVE_BURST_MIN_FRAMES = 8
    ACTIVE_BURST_MAX_FRAMES = 28
    ACTIVE_MIN_DETECTED_FRAMES = 6
    ACTIVE_MIN_FACE_FRAME_RATIO = 0.42
    ACTIVE_MIN_FACE_SIDE = 55
    ACTIVE_NOD_CY_RANGE = 0.052
    ACTIVE_SHAKE_CX_RANGE = 0.068
    ACTIVE_TURN_HEAD_DELTA = 0.038
    ACTIVE_TURN_PEAK_EDGE_MARGIN = 0.12

    LIVENESS_HINT = (
        "当前活体检测适用于课堂考勤辅助和内容安全风控预筛查，"
        "不应用于执法、金融开户或其他高风险身份决策。"
    )

    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
    DEEPFACE_HOME = DEEPFACE_HOME
    DEEPFACE_WEIGHT_DIR = os.path.join(DEEPFACE_HOME, ".deepface", "weights")
    LEGACY_DEEPFACE_WEIGHT_DIR = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
    DEEPFACE_MODEL_WEIGHT_FILES = {
        "Facenet": "facenet_weights.h5",
        "Facenet512": "facenet512_weights.h5",
    }
    DEEPFACE_MODEL_WEIGHT_URLS = {
        "Facenet": "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5",
        "Facenet512": "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
    }
    DEEPFACE_MODEL_DIMS = {
        "Facenet": 128,
        "Facenet512": 512,
        "InsightFace_buffalo_l": 512,
    }
    ANTISPOOF_REQUIRED_FILES = (
        "2.7_80x80_MiniFASNetV2.pth",
        "4_0_0_80x80_MiniFASNetV1SE.pth",
    )
    ANTISPOOF_WEIGHT_URLS = {
        "2.7_80x80_MiniFASNetV2.pth": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        "4_0_0_80x80_MiniFASNetV1SE.pth": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
    }

    SAFETY_BOUNDARY = [
        "仅限经授权的班级考勤、课堂活动统计与教学分析场景。",
        "不得用于持续监控、公开排名、歧视性决策或超出告知范围的人脸追踪。",
        "情绪分析仅作群体趋势参考，不应直接用于处分、心理诊断或成绩评价。",
        "建议启用最小化采集、访问鉴权、日志审计和数据定期清理策略。",
    ]

    @staticmethod
    def init_directories():
        for path in (
            Config.FACE_DB_PATH,
            Config.SNAPSHOT_PATH,
            Config.UPLOAD_FOLDER,
            os.path.dirname(Config.DATABASE_PATH),
        ):
            os.makedirs(path, exist_ok=True)


LIVENESS_METHODS = {
    "active": {
        "key": "active",
        "name": "随机动作活体",
        "description": "服务端随机下发点头/摇头/左右转头指令，浏览器连拍多帧做人脸框轨迹校验，并结合单帧静态筛查。",
    },
    "passive": {
        "key": "passive",
        "name": "本地活体检测",
        "description": "基于脸部纹理、细节熵、亮度分布和屏拍风险特征做单帧筛查。",
    },
}
