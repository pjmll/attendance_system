const TimeUtils = {
    formatTime(value) {
        if (!value) return '-';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return value;
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    },
    today() {
        return new Date().toISOString().split('T')[0];
    }
};

const FileUtils = {
    readAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },
    exportBlob(blob, filename) {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();
        URL.revokeObjectURL(link.href);
    }
};

const CameraUtils = {
    supported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    },
    async open(video, constraints) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
        video.srcObject = stream;
        return stream;
    },
    close(stream) {
        if (!stream) return;
        stream.getTracks().forEach(track => track.stop());
    },
    capture(video, canvas, quality = 0.88) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', quality);
    },
    async captureBurst(video, canvas, options = {}) {
        const durationMs = options.durationMs || 2800;
        const intervalMs = options.intervalMs || 160;
        const quality =
            options.quality !== undefined && options.quality !== null ? options.quality : 0.76;
        const frames = [];
        const start = Date.now();
        while (Date.now() - start < durationMs) {
            frames.push(this.capture(video, canvas, quality));
            await new Promise(resolve => setTimeout(resolve, intervalMs));
        }
        return frames;
    }
};

const ApiUtils = {
    create(baseURL) {
        return axios.create({
            baseURL,
            timeout: 45000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }
};

window.TimeUtils = TimeUtils;
window.FileUtils = FileUtils;
window.CameraUtils = CameraUtils;
window.ApiUtils = ApiUtils;
