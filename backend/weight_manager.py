import os
import shutil

import requests

try:
    from config import Config
except ModuleNotFoundError:
    from backend.config import Config


class WeightManager:
    def __init__(self):
        os.makedirs(Config.DEEPFACE_WEIGHT_DIR, exist_ok=True)

    @staticmethod
    def _file_exists(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    def _project_weight_path(self, filename):
        return os.path.join(Config.DEEPFACE_WEIGHT_DIR, filename)

    def _legacy_weight_path(self, filename):
        return os.path.join(Config.LEGACY_DEEPFACE_WEIGHT_DIR, filename)

    def ensure_local_copy(self, filename):
        target = self._project_weight_path(filename)
        if self._file_exists(target):
            return target, False

        legacy = self._legacy_weight_path(filename)
        if self._file_exists(legacy):
            shutil.copy2(legacy, target)
            return target, True
        return target, False

    def download_file(self, url, filename, timeout=60):
        target = self._project_weight_path(filename)
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(target, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 512):
                if chunk:
                    file_obj.write(chunk)
        return target

    def ensure_model_weight(self, model_name, allow_download=True):
        filename = Config.DEEPFACE_MODEL_WEIGHT_FILES.get(model_name)
        if not filename:
            return None, False
        target, copied = self.ensure_local_copy(filename)
        if self._file_exists(target):
            return target, True

        url = Config.DEEPFACE_MODEL_WEIGHT_URLS.get(model_name)
        if url and allow_download:
            try:
                self.download_file(url, filename, timeout=120)
            except Exception as exc:
                print(f"Download model weight failed for {model_name}: {exc}")
        return target, self._file_exists(target) or copied

    def ensure_antispoof_weights(self, allow_download=True):
        results = []
        for filename in Config.ANTISPOOF_REQUIRED_FILES:
            target, copied = self.ensure_local_copy(filename)
            if not self._file_exists(target):
                url = Config.ANTISPOOF_WEIGHT_URLS.get(filename)
                if url and allow_download:
                    try:
                        self.download_file(url, filename, timeout=120)
                    except Exception as exc:
                        print(f"Download anti-spoof weight failed for {filename}: {exc}")
            results.append(
                {
                    "filename": filename,
                    "path": target,
                    "available": self._file_exists(target),
                    "copied_from_legacy": copied,
                }
            )
        return results

    def ensure_startup_weights(self, model_name, allow_download=None):
        if allow_download is None:
            allow_download = Config.AUTO_DOWNLOAD_WEIGHTS_ON_STARTUP
        model_path, model_available = self.ensure_model_weight(model_name, allow_download=allow_download)
        antispoof = self.ensure_antispoof_weights(allow_download=allow_download)
        return {
            "model": {"name": model_name, "path": model_path, "available": model_available},
            "antispoof": antispoof,
            "auto_download": allow_download,
        }

    def list_status(self):
        model_status = {}
        for model_name, filename in Config.DEEPFACE_MODEL_WEIGHT_FILES.items():
            target = self._project_weight_path(filename)
            model_status[model_name] = {
                "filename": filename,
                "path": target,
                "available": self._file_exists(target),
            }

        antispoof_status = []
        for filename in Config.ANTISPOOF_REQUIRED_FILES:
            target = self._project_weight_path(filename)
            antispoof_status.append(
                {
                    "filename": filename,
                    "path": target,
                    "available": self._file_exists(target),
                }
            )
        return {"models": model_status, "antispoof": antispoof_status}
