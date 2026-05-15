new Vue({
    el: '#app',
    data() {
        return {
            apiBaseUrl: 'http://127.0.0.1:5001/api',
            api: null,
            loading: false,
            batchUploading: false,
            isAuthenticated: false,
            currentRole: 'teacher',
            activeTab: 'attendance',
            currentTimeText: '',
            clockTimer: null,
            loginForm: {
                username: '',
                password: '',
                role: 'teacher'
            },
            systemStatus: {
                face_database_count: 0,
                supported_liveness_methods: []
            },
            livenessMethods: [],
            selectedLivenessMethod: 'active',
            cameraStream: null,
            cameraReady: false,
            studentCameraStream: null,
            studentCameraReady: false,
            selectedDate: TimeUtils.today(),
            attendanceSummary: {
                total_students: 0,
                present_students: 0,
                absent_students: 0,
                total_records: 0,
                avg_confidence: 0,
                avg_liveness_score: 0
            },
            attendanceRecords: [],
            emotionStats: [],
            activityStats: {
                summary: {},
                rankings: [],
                students: [],
                activities: []
            },
            groupReports: [],
            students: [],
            lastAttendanceResult: null,
            attendanceChallenge: null,
            attendanceCountdown: 0,
            attendanceBurstBusy: false,
            groupResult: null,
            groupImageBase64: '',
            groupPreviewUrl: '',
            groupForm: {
                activity_name: '课堂点名',
                major: '不限',
                gender: '不限'
            },
            studentInputMode: 'camera',
            newStudent: {
                student_id: '',
                name: '',
                major: '',
                gender: '男',
                image: '',
                upload_mode: 'camera'
            },
            studentPreviewUrl: '',
            batchFiles: [],
            batchResults: null,
            studentSearchQuery: '',
            studentCurrentPage: 1,
            studentPageSize: 15,
            attendanceRecordPage: 1,
            attendanceRecordPageSize: 5,
            rankingPage: 1,
            rankingPageSize: 5,
            activityPage: 1,
            activityPageSize: 5,
            reportRecordPage: 1,
            reportRecordPageSize: 10
        };
    },
    computed: {
        availableTabs() {
            if (this.currentRole === 'teacher') {
                return [
                    { key: 'attendance', label: '实时考勤', icon: '◎' },
                    { key: 'group', label: '合照识别', icon: '群' },
                    { key: 'students', label: '学生管理', icon: '册' },
                    { key: 'reports', label: '统计报表', icon: '表' }
                ];
            }
            return [
                { key: 'attendance', label: '实时考勤', icon: '◎' },
                { key: 'group', label: '合照识别', icon: '群' },
                { key: 'personal', label: '个人考勤', icon: '人' }
            ];
        },
        userProfile() {
            const matchedStudent = this.findStudentByLogin();
            if (this.currentRole === 'teacher') {
                return {
                    name: this.loginForm.username || '课堂管理员',
                    id: 'ROLE-TEACHER',
                    department: '教学管理中心'
                };
            }
            if (matchedStudent) {
                return {
                    name: matchedStudent.name,
                    id: matchedStudent.student_id,
                    department: matchedStudent.major || '学生账号'
                };
            }
            return {
                name: this.loginForm.username || '学生用户',
                id: 'ROLE-STUDENT',
                department: '未匹配学生档案'
            };
        },
        userInitial() {
            return (this.userProfile.name || 'U').slice(0, 1).toUpperCase();
        },
        recentAttendanceRecords() {
            return this.attendanceRecords || [];
        },
        paginatedRecentAttendanceRecords() {
            const start = (this.attendanceRecordPage - 1) * this.attendanceRecordPageSize;
            return this.recentAttendanceRecords.slice(start, start + this.attendanceRecordPageSize);
        },
        recentAttendanceTotalPages() {
            return Math.ceil(this.recentAttendanceRecords.length / this.attendanceRecordPageSize) || 1;
        },
        reportStats() {
            const totalRecords = this.attendanceSummary.total_records || 0;
            const presentStudents = this.attendanceSummary.present_students || 0;
            const successCount = (this.attendanceRecords || []).filter(record => (record.status || '').toLowerCase() !== 'failed').length;
            const failureCount = Math.max(totalRecords - successCount, 0);
            const happyItem = (this.emotionStats || []).find(item => item.name === 'happy');
            const emotionTotal = (this.emotionStats || []).reduce((sum, item) => sum + (item.value || 0), 0);
            const happyEmotionRate = emotionTotal
                ? Math.round((((happyItem && happyItem.value) || 0) / emotionTotal) * 100)
                : 0;
            const groupActivityCount = (this.groupReports || []).length;

            return {
                todayAttendanceTotal: presentStudents,
                successCount,
                failureCount,
                attendanceSuccessRate: totalRecords ? Math.round((successCount / totalRecords) * 100) : 0,
                failureRate: totalRecords ? Math.round((failureCount / totalRecords) * 100) : 0,
                happyEmotionRate,
                groupActivityCount,
                emotionTotal
            };
        },
        emotionBreakdown() {
            const total = (this.emotionStats || []).reduce((sum, item) => sum + (item.value || 0), 0);
            const colors = {
                happy: '#d17961',
                neutral: '#8a8f9f',
                focused: '#376f6b',
                sad: '#7e91be',
                angry: '#b45252',
                surprise: '#d6a14a',
                fear: '#8b6ea9',
                disgust: '#68835d'
            };

            return (this.emotionStats || []).map(item => ({
                name: item.name,
                label: this.formatEmotion(item.name),
                value: item.value || 0,
                percent: total ? Math.round(((item.value || 0) / total) * 100) : 0,
                color: colors[item.name] || '#b36a5e'
            }));
        },
        ringBackgroundStyle() {
            const items = this.emotionBreakdown.filter(item => item.percent > 0);
            if (!items.length) {
                return { background: '#dfe6e0' };
            }
            let cumulative = 0;
            const segments = [];
            items.forEach((item, i) => {
                const start = cumulative;
                cumulative += item.percent;
                const end = i === items.length - 1 ? 100 : cumulative;
                segments.push(`${item.color} ${start}% ${end}%`);
            });
            return {
                background: `conic-gradient(from -90deg, ${segments.join(', ')})`
            };
        },
        filteredStudents() {
            const q = (this.studentSearchQuery || '').trim().toLowerCase();
            if (!q) return this.students || [];
            return (this.students || []).filter(s =>
                (s.student_id || '').toLowerCase().includes(q) ||
                (s.name || '').toLowerCase().includes(q)
            );
        },
        paginatedStudents() {
            const start = (this.studentCurrentPage - 1) * this.studentPageSize;
            return this.filteredStudents.slice(start, start + this.studentPageSize);
        },
        studentTotalPages() {
            return Math.ceil(this.filteredStudents.length / this.studentPageSize) || 1;
        },
        majorStats() {
            const countedStudents = new Set();
            const counter = {};
            (this.attendanceRecords || []).forEach(record => {
                if (!countedStudents.has(record.student_id)) {
                    countedStudents.add(record.student_id);
                    const name = record.class_name || '未设置专业';
                    counter[name] = (counter[name] || 0) + 1;
                }
            });
            const total = Object.values(counter).reduce((sum, value) => sum + value, 0);
            return Object.keys(counter).map(name => ({
                name,
                count: counter[name],
                percent: total ? Math.round((counter[name] / total) * 100) : 0
            })).sort((left, right) => right.count - left.count);
        },
        studentRanking() {
            return this.activityStats.rankings || [];
        },
        paginatedStudentRanking() {
            const start = (this.rankingPage - 1) * this.rankingPageSize;
            return this.studentRanking.slice(start, start + this.rankingPageSize);
        },
        rankingTotalPages() {
            return Math.ceil(this.studentRanking.length / this.rankingPageSize) || 1;
        },
        activityCards() {
            return (this.groupReports || []).map(item => {
                const rate = item.total_faces ? Math.round(((item.matched_faces || 0) / item.total_faces) * 100) : 0;
                return {
                    id: item.id,
                    activity_name: item.activity_name || '未命名活动',
                    created_at: item.created_at,
                    total_faces: item.total_faces || 0,
                    matched_faces: item.matched_faces || 0,
                    recognitionRate: rate
                };
            });
        },
        paginatedActivityCards() {
            const start = (this.activityPage - 1) * this.activityPageSize;
            return this.activityCards.slice(start, start + this.activityPageSize);
        },
        activityTotalPages() {
            return Math.ceil(this.activityCards.length / this.activityPageSize) || 1;
        },
        paginatedReportRecords() {
            const start = (this.reportRecordPage - 1) * this.reportRecordPageSize;
            return (this.attendanceRecords || []).slice(start, start + this.reportRecordPageSize);
        },
        reportRecordTotalPages() {
            return Math.ceil((this.attendanceRecords || []).length / this.reportRecordPageSize) || 1;
        },
        personalAttendanceRecords() {
            if (this.currentRole !== 'student') {
                return [];
            }
            const matchedStudent = this.findStudentByLogin();
            const username = (this.loginForm.username || '').trim();
            return (this.attendanceRecords || []).filter(record => {
                if (matchedStudent) {
                    return record.student_id === matchedStudent.student_id;
                }
                return record.student_id === username || record.student_name === username;
            });
        },
        studentLinkedRecordCount() {
            return this.personalAttendanceRecords.length;
        },
        studentLinkedActivityCount() {
            if (this.currentRole !== 'student') {
                return 0;
            }
            const matchedStudent = this.findStudentByLogin();
            const username = (this.loginForm.username || '').trim();
            const studentEntry = (this.activityStats.students || []).find(item => {
                if (matchedStudent) {
                    return item.student_id === matchedStudent.student_id;
                }
                return item.student_id === username || item.student_name === username;
            });
            return studentEntry ? studentEntry.participation_count || 0 : 0;
        }
    },
    watch: {
        filteredStudents() {
            if (this.studentCurrentPage > this.studentTotalPages) {
                this.studentCurrentPage = Math.max(1, this.studentTotalPages);
            }
        },
        recentAttendanceRecords() {
            if (this.attendanceRecordPage > this.recentAttendanceTotalPages) {
                this.attendanceRecordPage = Math.max(1, this.recentAttendanceTotalPages);
            }
        },
        studentRanking() {
            if (this.rankingPage > this.rankingTotalPages) {
                this.rankingPage = Math.max(1, this.rankingTotalPages);
            }
        },
        activityCards() {
            if (this.activityPage > this.activityTotalPages) {
                this.activityPage = Math.max(1, this.activityTotalPages);
            }
        },
        attendanceRecords() {
            if (this.reportRecordPage > this.reportRecordTotalPages) {
                this.reportRecordPage = Math.max(1, this.reportRecordTotalPages);
            }
        }
    },
    mounted() {
        this.api = ApiUtils.create(this.apiBaseUrl);
        this.updateClock();
        this.clockTimer = window.setInterval(this.updateClock, 1000);
        this.bootstrap();
    },
    beforeDestroy() {
        this.stopAttendanceCamera();
        this.stopStudentCamera();
        if (this.clockTimer) {
            window.clearInterval(this.clockTimer);
        }
    },
    methods: {
        async bootstrap() {
            await this.refreshDashboardData();
        },
        async refreshDashboardData() {
            const tasks = [
                this.fetchSystemStatus(),
                this.fetchLivenessMethods(),
                this.fetchStudents(),
                this.fetchAttendanceSummary(),
                this.fetchAttendanceRecords(),
                this.fetchEmotionStats(),
                this.fetchActivityStats(),
                this.fetchGroupReports()
            ];
            const results = await Promise.allSettled(tasks);
            const failedCount = results.filter(item => item.status === 'rejected').length;
            if (failedCount && this.isAuthenticated) {
                this.$message.warning(`有 ${failedCount} 项数据未加载成功，可稍后刷新`);
            }
        },
        async fetchSystemStatus() {
            const { data } = await this.api.get('/system_status');
            this.systemStatus = data || this.systemStatus;
        },
        async fetchLivenessMethods() {
            const { data } = await this.api.get('/liveness_methods');
            this.livenessMethods = data.methods || [];
            this.selectedLivenessMethod = data.default_method || 'active';
        },
        async fetchStudents() {
            const { data } = await this.api.get('/students');
            this.students = data.students || [];
        },
        async fetchAttendanceSummary() {
            const { data } = await this.api.get('/attendance_summary', {
                params: { date: this.selectedDate }
            });
            this.attendanceSummary = data.summary || this.attendanceSummary;
        },
        async fetchAttendanceRecords() {
            const { data } = await this.api.get('/attendance_records', {
                params: { date: this.selectedDate }
            });
            this.attendanceRecords = data.records || [];
        },
        async fetchEmotionStats() {
            const { data } = await this.api.get('/emotion_stats', {
                params: { date: this.selectedDate }
            });
            this.emotionStats = data.data || [];
        },
        async fetchActivityStats() {
            const { data } = await this.api.get('/activity_stats', {
                params: { date: this.selectedDate, limit: 12 }
            });
            this.activityStats = {
                summary: data.summary || {},
                rankings: data.rankings || [],
                students: data.students || [],
                activities: data.activities || []
            };
        },
        async fetchGroupReports() {
            const { data } = await this.api.get('/group_reports');
            this.groupReports = data.reports || [];
        },
        updateClock() {
            this.currentTimeText = TimeUtils.formatTime(new Date());
        },
        handleLogin() {
            if (!this.loginForm.username || !this.loginForm.password) {
                try {
                    if (this.$message && this.$message.warning) {
                        this.$message.warning('请输入用户名和密码');
                    } else {
                        window.alert('请输入用户名和密码');
                    }
                } catch (_e) {
                    window.alert('请输入用户名和密码');
                }
                return;
            }
            this.isAuthenticated = true;
            this.currentRole = this.loginForm.role;
            this.activeTab = 'attendance';
            const tip = this.currentRole === 'teacher' ? '已进入教师模式' : '已进入学生模式';
            try {
                if (this.$message && this.$message.success) {
                    this.$message.success(tip);
                } else {
                    window.alert(tip);
                }
            } catch (_e) {
                window.alert(tip);
            }
        },
        handleLogout() {
            this.stopAttendanceCamera();
            this.stopStudentCamera();
            this.isAuthenticated = false;
            this.currentRole = this.loginForm.role;
            this.activeTab = 'attendance';
            this.lastAttendanceResult = null;
            this.groupResult = null;
        },
        async startAttendanceCamera() {
            if (!CameraUtils.supported()) {
                this.$message.error('当前浏览器不支持摄像头访问');
                return;
            }
            this.lastAttendanceResult = null;
            try {
                this.cameraStream = await CameraUtils.open(this.$refs.attendanceVideo, {
                    width: 640,
                    height: 480,
                    facingMode: 'user'
                });
                this.cameraReady = true;
            } catch (_error) {
                this.$message.error('摄像头启动失败，请检查浏览器权限');
            }
        },
        stopAttendanceCamera() {
            CameraUtils.close(this.cameraStream);
            this.cameraStream = null;
            this.cameraReady = false;
        },
        async startStudentCamera() {
            if (!CameraUtils.supported()) {
                this.$message.error('当前浏览器不支持摄像头访问');
                return;
            }
            try {
                this.studentCameraStream = await CameraUtils.open(this.$refs.studentVideo, {
                    width: 480,
                    height: 360,
                    facingMode: 'user'
                });
                this.studentCameraReady = true;
            } catch (_error) {
                this.$message.error('学生采集摄像头启动失败');
            }
        },
        stopStudentCamera() {
            CameraUtils.close(this.studentCameraStream);
            this.studentCameraStream = null;
            this.studentCameraReady = false;
        },
        captureAttendanceFrame() {
            if (!this.cameraReady) {
                this.$message.warning('请先启动摄像头');
                return '';
            }
            return CameraUtils.capture(this.$refs.attendanceVideo, this.$refs.attendanceCanvas);
        },
        captureStudentFrame() {
            if (!this.studentCameraReady) {
                this.$message.warning('请先启动学生采集摄像头');
                return;
            }
            this.newStudent.image = CameraUtils.capture(this.$refs.studentVideo, this.$refs.studentCanvas);
            this.newStudent.upload_mode = 'camera';
            this.studentPreviewUrl = this.newStudent.image;
            this.$message.success('已采集学生人脸照片');
        },
        async onStudentFileChange(event) {
            const file = event.target.files && event.target.files[0];
            if (!file) return;
            this.newStudent.image = await FileUtils.readAsBase64(file);
            this.newStudent.upload_mode = 'upload';
            this.studentPreviewUrl = this.newStudent.image;
            event.target.value = '';
            this.$message.success('已载入本地照片');
        },
        switchStudentInputMode(mode) {
            this.studentInputMode = mode;
            this.newStudent.upload_mode = mode;
            this.newStudent.image = '';
            this.studentPreviewUrl = '';
            if (mode !== 'camera') {
                this.stopStudentCamera();
            }
        },
        async submitAttendance() {
            if (this.selectedLivenessMethod === 'active') {
                return this.submitAttendanceActive();
            }
            const image = this.captureAttendanceFrame();
            if (!image) return;

            this.loading = true;
            try {
                const { data } = await this.api.post('/attendance', {
                    image,
                    liveness_method: this.selectedLivenessMethod
                });
                this.lastAttendanceResult = data;
                this.$message.success(data.message || '考勤成功');
                await Promise.all([
                    this.fetchAttendanceSummary(),
                    this.fetchAttendanceRecords(),
                    this.fetchEmotionStats(),
                    this.fetchActivityStats()
                ]);
            } catch (error) {
                if (error.code === 'ECONNABORTED') {
                    this.lastAttendanceResult = {
                        success: false,
                        error: '识别请求超时，请重新拍摄后再试'
                    };
                    this.$message.error('识别请求超时，请稍后重试');
                } else {
                    const payload = (error.response && error.response.data) || {};
                    this.lastAttendanceResult = {
                        success: false,
                        error: payload.error || '考勤请求失败',
                        hint: payload.message || ''
                    };
                    this.$message.error(payload.error || '考勤请求失败');
                }
            } finally {
                this.loading = false;
            }
        },
        sleepMs(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        },
        async submitAttendanceActive() {
            if (!this.cameraReady) {
                this.$message.warning('请先启动摄像头');
                return;
            }
            this.lastAttendanceResult = null;
            this.loading = true;
            let challengeId = '';
            try {
                const chRes = await this.api.post('/liveness_challenge', {});
                const ch = chRes.data || {};
                challengeId = ch.challenge_id || '';
                if (!challengeId) {
                    throw new Error((ch && ch.error) || '未获取到动作挑战');
                }
                this.attendanceChallenge = { prompt: ch.prompt || '请按提示完成动作' };
                await this.$nextTick();
                for (let c = 3; c >= 1; c -= 1) {
                    this.attendanceCountdown = c;
                    await this.sleepMs(900);
                }
                this.attendanceCountdown = 0;
                this.attendanceBurstBusy = true;
                const frames = await CameraUtils.captureBurst(
                    this.$refs.attendanceVideo,
                    this.$refs.attendanceCanvas,
                    { durationMs: 2800, intervalMs: 160, quality: 0.76 }
                );
                this.attendanceBurstBusy = false;
                this.attendanceChallenge = null;
                if (!frames.length) {
                    this.$message.warning('未采集到画面，请重试');
                    return;
                }
                const image = frames[frames.length - 1];
                const { data } = await this.api.post(
                    '/attendance',
                    {
                        image,
                        challenge_id: challengeId,
                        action_frames: frames,
                        liveness_method: 'active'
                    },
                    { timeout: 120000 }
                );
                this.lastAttendanceResult = data;
                this.$message.success(data.message || '考勤成功');
                await Promise.all([
                    this.fetchAttendanceSummary(),
                    this.fetchAttendanceRecords(),
                    this.fetchEmotionStats(),
                    this.fetchActivityStats()
                ]);
            } catch (error) {
                this.attendanceChallenge = null;
                this.attendanceCountdown = 0;
                this.attendanceBurstBusy = false;
                if (error.code === 'ECONNABORTED') {
                    this.lastAttendanceResult = {
                        success: false,
                        error: '识别请求超时，请重新考勤'
                    };
                    this.$message.error('识别请求超时，请稍后重试');
                } else if (error.message && !error.response) {
                    this.$message.error(error.message);
                } else {
                    const payload = (error.response && error.response.data) || {};
                    this.lastAttendanceResult = {
                        success: false,
                        error: payload.error || '考勤请求失败',
                        hint: payload.message || ''
                    };
                    this.$message.error(payload.error || '考勤请求失败');
                }
            } finally {
                this.loading = false;
                this.attendanceBurstBusy = false;
                this.attendanceCountdown = 0;
                this.attendanceChallenge = null;
            }
        },
        async submitStudent() {
            if (!this.newStudent.student_id || !this.newStudent.name || !this.newStudent.major) {
                this.$message.warning('请完整填写学生信息');
                return;
            }
            if (!this.newStudent.image) {
                this.$message.warning('请先采集或上传学生人脸照片');
                return;
            }

            this.loading = true;
            try {
                const { data } = await this.api.post('/add_student', this.newStudent);
                this.$message.success(data.message || '学生录入成功');
                this.newStudent = {
                    student_id: '',
                    name: '',
                    major: '',
                    gender: '男',
                    image: '',
                    upload_mode: this.studentInputMode
                };
                this.studentPreviewUrl = '';
                await Promise.all([this.fetchStudents(), this.fetchSystemStatus()]);
            } catch (error) {
                this.$message.error(
                    ((error.response && error.response.data && error.response.data.error) || '学生录入失败')
                );
            } finally {
                this.loading = false;
            }
        },
        async deleteStudent(student) {
            try {
                await this.$confirm(`确认删除 ${student.name} (${student.student_id}) 吗？`, '删除学生', {
                    type: 'warning'
                });
                await this.api.delete(`/delete_student/${student.student_id}`);
                this.$message.success('学生已删除');
                await Promise.all([this.fetchStudents(), this.fetchSystemStatus()]);
            } catch (error) {
                if (error && error !== 'cancel' && error !== 'close') {
                    this.$message.error(
                        ((error.response && error.response.data && error.response.data.error) || '删除失败')
                    );
                }
            }
        },
        onBatchFileChange(event) {
            const files = Array.from(event.target.files || []);
            const maxSize = 12 * 1024 * 1024;
            const validFiles = [];
            files.forEach(file => {
                if (file.size > maxSize) {
                    this.$message.warning(`文件 "${file.name}" 超过 12MB，已自动跳过`);
                } else {
                    validFiles.push(file);
                }
            });
            this.batchFiles = this.batchFiles.concat(validFiles);
            event.target.value = '';
        },
        removeBatchFile(index) {
            this.batchFiles.splice(index, 1);
        },
        async submitBatchUpload() {
            if (!this.batchFiles.length) {
                this.$message.warning('请先选择人脸照片文件');
                return;
            }
            this.batchUploading = true;
            try {
                const formData = new FormData();
                this.batchFiles.forEach(file => formData.append('files', file));
                const { data } = await this.api.post('/batch_upload_faces', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                this.batchResults = data;
                (data.results || []).forEach(item => {
                    if (item.success) {
                        this.$message.success(`${item.filename} · ${item.name || item.student_id} 录入成功`);
                    } else {
                        this.$message.warning(`${item.filename} 录入失败: ${item.error || '未知错误'}`);
                    }
                });
                this.batchFiles = [];
                this.$message.success(`批量上传完成：${data.success_count} 成功，${data.fail_count} 失败`);
                await Promise.all([this.fetchStudents(), this.fetchSystemStatus()]);
            } catch (error) {
                this.$message.error(
                    ((error.response && error.response.data && error.response.data.error) || '批量上传失败')
                );
            } finally {
                this.batchUploading = false;
            }
        },
        async onGroupFileChange(event) {
            const file = event.target.files && event.target.files[0];
            if (!file) return;
            this.groupImageBase64 = await FileUtils.readAsBase64(file);
            this.groupPreviewUrl = this.groupImageBase64;
            event.target.value = '';
        },
        async submitGroupRecognition() {
            if (!this.groupImageBase64) {
                this.$message.warning('请先上传合照');
                return;
            }

            this.loading = true;
            try {
                const { data } = await this.api.post('/group_recognition', {
                    image: this.groupImageBase64,
                    activity_name: this.groupForm.activity_name,
                    major: this.groupForm.major,
                    gender: this.groupForm.gender
                });
                this.groupResult = data;
                this.$message.success(`合照识别完成，匹配 ${data.matched_count} 人`);
                await Promise.all([
                    this.fetchAttendanceSummary(),
                    this.fetchAttendanceRecords(),
                    this.fetchEmotionStats(),
                    this.fetchActivityStats(),
                    this.fetchGroupReports()
                ]);
            } catch (error) {
                if (error.code === 'ECONNABORTED') {
                    this.$message.error('合照识别请求超时，请压缩图片后再试');
                } else {
                    this.$message.error(
                        ((error.response && error.response.data && error.response.data.error) || '合照识别失败')
                    );
                }
            } finally {
                this.loading = false;
            }
        },
        async exportAttendance() {
            try {
                const response = await this.api.get('/export_attendance', {
                    params: { date: this.selectedDate },
                    responseType: 'blob'
                });
                FileUtils.exportBlob(response.data, `attendance_${this.selectedDate}.csv`);
            } catch (_error) {
                this.$message.error('导出报表失败');
            }
        },
        async handleDateChange() {
            await Promise.all([
                this.fetchAttendanceSummary(),
                this.fetchAttendanceRecords(),
                this.fetchEmotionStats(),
                this.fetchActivityStats()
            ]);
        },
        onStudentSearchInput() {
            this.studentCurrentPage = 1;
        },
        goToStudentPage(page) {
            if (page >= 1 && page <= this.studentTotalPages) {
                this.studentCurrentPage = page;
            }
        },
        goToAttendancePage(page) {
            if (page >= 1 && page <= this.recentAttendanceTotalPages) {
                this.attendanceRecordPage = page;
            }
        },
        goToRankingPage(page) {
            if (page >= 1 && page <= this.rankingTotalPages) {
                this.rankingPage = page;
            }
        },
        goToActivityPage(page) {
            if (page >= 1 && page <= this.activityTotalPages) {
                this.activityPage = page;
            }
        },
        goToReportRecordPage(page) {
            if (page >= 1 && page <= this.reportRecordTotalPages) {
                this.reportRecordPage = page;
            }
        },
        findStudentByLogin() {
            const username = (this.loginForm.username || '').trim();
            return (this.students || []).find(student => student.student_id === username || student.name === username) || null;
        },
        formatEmotion(value) {
            const map = {
                neutral: '平静',
                happy: '开心',
                angry: '愤怒',
                sad: '低落',
                fear: '紧张',
                disgust: '厌恶',
                surprise: '惊讶',
                focused: '专注'
            };
            return map[value] || value || '-';
        },
        formatTime(value) {
            return TimeUtils.formatTime(value);
        }
    }
});