import csv
import io
import json
import os
import sqlite3
from collections import Counter

import numpy as np


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                major TEXT DEFAULT '',
                gender TEXT DEFAULT '',
                face_encoding TEXT,
                face_model TEXT,
                face_embedding_dim INTEGER,
                photo_path TEXT,
                face_quality_score REAL DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute("PRAGMA table_info(students)")
        student_columns = [row[1] for row in cursor.fetchall()]
        if "face_model" not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN face_model TEXT")
        if "face_embedding_dim" not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN face_embedding_dim INTEGER")
        if "major" not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN major TEXT DEFAULT ''")
        if "gender" not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN gender TEXT DEFAULT ''")
        if "class_name" in student_columns:
            self._migrate_drop_class_name(cursor, conn)

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                class_name TEXT,
                check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detection_method TEXT NOT NULL,
                confidence REAL DEFAULT 0,
                liveness_score REAL DEFAULT 0,
                status TEXT DEFAULT 'present',
                emotion TEXT DEFAULT '平静',
                snapshot_path TEXT,
                source TEXT DEFAULT 'attendance',
                activity_name TEXT DEFAULT ''
            )
            """
        )

        cursor.execute("PRAGMA table_info(attendance_records)")
        attendance_columns = [row[1] for row in cursor.fetchall()]
        if "activity_name" not in attendance_columns:
            cursor.execute("ALTER TABLE attendance_records ADD COLUMN activity_name TEXT DEFAULT ''")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS group_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_name TEXT NOT NULL,
                class_name TEXT,
                total_faces INTEGER DEFAULT 0,
                matched_faces INTEGER DEFAULT 0,
                unknown_faces INTEGER DEFAULT 0,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS group_session_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT,
                student_name TEXT,
                emotion TEXT DEFAULT '平静',
                confidence REAL DEFAULT 0,
                face_box TEXT,
                is_matched INTEGER DEFAULT 0,
                FOREIGN KEY(session_id) REFERENCES group_sessions(id)
            )
            """
        )

        conn.commit()
        conn.close()

    def _migrate_drop_class_name(self, cursor, conn):
        cursor.execute("""
            CREATE TABLE students_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                major TEXT DEFAULT '',
                gender TEXT DEFAULT '',
                face_encoding TEXT,
                face_model TEXT,
                face_embedding_dim INTEGER,
                photo_path TEXT,
                face_quality_score REAL DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO students_new
            (id, student_id, name, major, gender, face_encoding, face_model,
             face_embedding_dim, photo_path, face_quality_score, is_active,
             created_at, updated_at)
            SELECT id, student_id, name,
                   COALESCE(major, ''), COALESCE(gender, ''),
                   face_encoding, face_model, face_embedding_dim,
                   photo_path, face_quality_score, is_active,
                   created_at, updated_at
            FROM students
        """)
        cursor.execute("DROP TABLE students")
        cursor.execute("ALTER TABLE students_new RENAME TO students")
        conn.commit()

    @staticmethod
    def face_encoding_to_string(face_encoding):
        if face_encoding is None:
            return None
        return json.dumps(np.asarray(face_encoding, dtype=np.float32).tolist())

    @staticmethod
    def string_to_face_encoding(encoding_string):
        if not encoding_string:
            return None
        return np.array(json.loads(encoding_string), dtype=np.float32)

    def add_student(self, student_id, name, major, gender, photo_path=None):
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO students (student_id, name, major, gender, photo_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, name, major, gender, photo_path),
            )
            conn.commit()
            return True, "学生添加成功"
        except sqlite3.IntegrityError:
            return False, "学号已存在"
        finally:
            conn.close()

    def upsert_student(self, student_id, name, major, gender, photo_path=None):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO students (student_id, name, major, gender, photo_path)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                name = excluded.name,
                major = excluded.major,
                gender = excluded.gender,
                photo_path = excluded.photo_path,
                updated_at = CURRENT_TIMESTAMP
            """,
            (student_id, name, major, gender, photo_path),
        )
        conn.commit()
        conn.close()

    def get_student(self, student_id):
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ? AND is_active = 1",
            (student_id,),
        ).fetchone()
        conn.close()
        return row

    def get_all_students(self):
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM students
            WHERE is_active = 1
            ORDER BY student_id
            """
        ).fetchall()
        conn.close()
        return rows

    def update_student_face_encoding(self, student_id, face_encoding, face_quality_score=0.0, face_model=None):
        conn = self._connect()
        cursor = conn.cursor()
        embedding_dim = int(len(face_encoding)) if face_encoding is not None else None
        cursor.execute(
            """
            UPDATE students
            SET face_encoding = ?, face_quality_score = ?, face_model = ?, face_embedding_dim = ?, updated_at = CURRENT_TIMESTAMP
            WHERE student_id = ?
            """,
            (
                self.face_encoding_to_string(face_encoding),
                face_quality_score,
                face_model,
                embedding_dim,
                student_id,
            ),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated

    def get_all_face_encodings(self):
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT student_id, name, major, gender, face_encoding
                 , face_model, face_embedding_dim
                 , photo_path
            FROM students
            WHERE is_active = 1 AND face_encoding IS NOT NULL
            """
        ).fetchall()
        conn.close()

        records = []
        for row in rows:
            encoding = self.string_to_face_encoding(row["face_encoding"])
            if encoding is not None:
                records.append(
                    {
                        "student_id": row["student_id"],
                        "name": row["name"],
                        "major": row["major"],
                        "gender": row["gender"] or "",
                        "encoding": encoding,
                        "face_model": row["face_model"],
                        "face_embedding_dim": row["face_embedding_dim"],
                        "photo_path": row["photo_path"],
                    }
                )
        return records

    def delete_student(self, student_id):
        conn = self._connect()
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT photo_path FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        photo_path = row["photo_path"] if row else None
        cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        if deleted and photo_path:
            try:
                if os.path.isfile(photo_path):
                    os.remove(photo_path)
            except OSError:
                pass
        return deleted, "学生删除成功" if deleted else "学生不存在"

    def add_attendance_record(
        self,
        student_id,
        student_name,
        class_name,
        detection_method,
        confidence,
        liveness_score,
        emotion,
        snapshot_path=None,
        source="attendance",
        activity_name="",
    ):
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO attendance_records
            (student_id, student_name, class_name, detection_method, confidence,
             liveness_score, emotion, snapshot_path, source, activity_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                student_name,
                class_name,
                detection_method,
                confidence,
                liveness_score,
                emotion,
                snapshot_path,
                source,
                activity_name,
            ),
        )
        conn.commit()
        conn.close()

    def add_group_session(self, activity_name, class_name, total_faces, matched_faces, image_path, results):
        unknown_faces = max(total_faces - matched_faces, 0)
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO group_sessions
            (activity_name, class_name, total_faces, matched_faces, unknown_faces, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (activity_name, class_name, total_faces, matched_faces, unknown_faces, image_path),
        )
        session_id = cursor.lastrowid

        for item in results:
            cursor.execute(
                """
                INSERT INTO group_session_members
                (session_id, student_id, student_name, emotion, confidence, face_box, is_matched)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    item.get("student_id"),
                    item.get("student_name"),
                    item.get("emotion"),
                    item.get("confidence", 0),
                    json.dumps(item.get("face_location", {}), ensure_ascii=False),
                    1 if item.get("matched") else 0,
                ),
            )
        conn.commit()
        conn.close()
        return session_id

    def get_today_attendance(self):
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM attendance_records
            WHERE DATE(check_time, 'localtime') = DATE('now', 'localtime')
            ORDER BY check_time DESC
            """
        ).fetchall()
        conn.close()
        return rows

    def get_attendance_by_date(self, date_str):
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM attendance_records
            WHERE DATE(check_time, 'localtime') = ?
            ORDER BY check_time DESC
            """,
            (date_str,),
        ).fetchall()
        conn.close()
        return rows

    def delete_attendance_records(self, date_str=None):
        conn = self._connect()
        try:
            if date_str:
                conn.execute(
                    "DELETE FROM attendance_records WHERE DATE(check_time, 'localtime') = ?",
                    (date_str,)
                )
            else:
                conn.execute("DELETE FROM attendance_records")
            conn.commit()
            return True, "删除成功"
        except sqlite3.Error as e:
            return False, str(e)
        finally:
            conn.close()

    def get_attendance_summary(self, date=None):
        conn = self._connect()
        if date:
            row = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT student_id) AS present_students,
                    COUNT(*) AS total_records,
                    AVG(confidence) AS avg_confidence,
                    AVG(liveness_score) AS avg_liveness_score
                FROM attendance_records
                WHERE DATE(check_time, 'localtime') = ?
                """,
                (date,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT student_id) AS present_students,
                    COUNT(*) AS total_records,
                    AVG(confidence) AS avg_confidence,
                    AVG(liveness_score) AS avg_liveness_score
                FROM attendance_records
                WHERE DATE(check_time, 'localtime') = DATE('now', 'localtime')
                """
            ).fetchone()

        total_students = conn.execute(
            "SELECT COUNT(*) AS total FROM students WHERE is_active = 1"
        ).fetchone()["total"]
        conn.close()

        present_students = row["present_students"] or 0
        absent_students = max(total_students - present_students, 0)
        return {
            "total_students": total_students,
            "present_students": present_students,
            "absent_students": absent_students,
            "total_records": row["total_records"] or 0,
            "avg_confidence": round(row["avg_confidence"] or 0, 4),
            "avg_liveness_score": round(row["avg_liveness_score"] or 0, 4),
        }

    def get_emotion_stats(self, date=None):
        conn = self._connect()
        params = ()
        query = "SELECT emotion FROM attendance_records"
        if date:
            query += " WHERE DATE(check_time, 'localtime') = ?"
            params = (date,)
        rows = conn.execute(query, params).fetchall()
        group_rows = conn.execute("SELECT emotion FROM group_session_members").fetchall()
        conn.close()

        counter = Counter()
        for row in rows:
            counter[row["emotion"] or "平静"] += 1
        for row in group_rows:
            counter[row["emotion"] or "平静"] += 1

        return [{"name": name, "value": value} for name, value in counter.most_common()]

    def get_group_reports(self):
        conn = self._connect()
        sessions = conn.execute(
            """
            SELECT * FROM group_sessions
            ORDER BY created_at DESC
            LIMIT 20
            """
        ).fetchall()
        conn.close()
        return [dict(row) for row in sessions]

    def get_activity_stats(self, date=None, limit=12):
        conn = self._connect()
        params = []
        date_filter = ""
        if date:
            date_filter = "WHERE DATE(gs.created_at, 'localtime') = ?"
            params.append(date)

        base_query = f"""
            WITH matched_members AS (
                SELECT DISTINCT
                    gm.session_id,
                    gm.student_id,
                    gm.student_name
                FROM group_session_members gm
                WHERE gm.is_matched = 1 AND gm.student_id IS NOT NULL AND gm.student_id != ''
            )
            SELECT
                mm.student_id,
                mm.student_name,
                COALESCE(s.major, gs.class_name, '') AS class_name,
                gs.activity_name,
                gs.created_at
            FROM matched_members mm
            JOIN group_sessions gs ON gs.id = mm.session_id
            LEFT JOIN students s ON s.student_id = mm.student_id
            {date_filter}
        """

        rows = conn.execute(base_query, tuple(params)).fetchall()

        ranking_query = f"""
            WITH matched_members AS (
                SELECT DISTINCT
                    gm.session_id,
                    gm.student_id,
                    gm.student_name
                FROM group_session_members gm
                WHERE gm.is_matched = 1 AND gm.student_id IS NOT NULL AND gm.student_id != ''
            )
            SELECT
                mm.student_id,
                mm.student_name,
                COALESCE(s.major, gs.class_name, '') AS class_name,
                COUNT(*) AS participation_count,
                COUNT(DISTINCT gs.activity_name) AS activity_type_count,
                MAX(gs.created_at) AS last_participation_time
            FROM matched_members mm
            JOIN group_sessions gs ON gs.id = mm.session_id
            LEFT JOIN students s ON s.student_id = mm.student_id
            {date_filter}
            GROUP BY mm.student_id, mm.student_name, COALESCE(s.major, gs.class_name, '')
            ORDER BY participation_count DESC, last_participation_time DESC, mm.student_id ASC
            LIMIT ?
        """
        ranking_rows = conn.execute(ranking_query, tuple(params + [limit])).fetchall()
        conn.close()

        student_stats = {}
        activity_summary = Counter()

        for row in rows:
            student_id = row["student_id"]
            activity_name = row["activity_name"] or "未命名活动"
            if student_id not in student_stats:
                student_stats[student_id] = {
                    "student_id": student_id,
                    "student_name": row["student_name"],
                    "class_name": row["class_name"] or "",
                    "participation_count": 0,
                    "activity_type_count": 0,
                    "last_participation_time": row["created_at"],
                    "activity_breakdown": Counter(),
                }

            student_stats[student_id]["participation_count"] += 1
            student_stats[student_id]["activity_breakdown"][activity_name] += 1
            activity_summary[activity_name] += 1

            if row["created_at"] and row["created_at"] > student_stats[student_id]["last_participation_time"]:
                student_stats[student_id]["last_participation_time"] = row["created_at"]

        students = []
        for item in student_stats.values():
            breakdown = [
                {"activity_name": name, "count": count}
                for name, count in item["activity_breakdown"].most_common()
            ]
            students.append(
                {
                    "student_id": item["student_id"],
                    "student_name": item["student_name"],
                    "class_name": item["class_name"],
                    "participation_count": item["participation_count"],
                    "activity_type_count": len(item["activity_breakdown"]),
                    "last_participation_time": item["last_participation_time"],
                    "activity_breakdown": breakdown,
                }
            )

        students.sort(
            key=lambda item: (
                -item["participation_count"],
                -(item["activity_type_count"] or 0),
                item["student_id"],
            )
        )

        rankings = [
            {
                "student_id": row["student_id"],
                "student_name": row["student_name"],
                "class_name": row["class_name"] or "",
                "participation_count": int(row["participation_count"] or 0),
                "activity_type_count": int(row["activity_type_count"] or 0),
                "last_participation_time": row["last_participation_time"],
            }
            for row in ranking_rows
        ]

        activities = [
            {"activity_name": name, "participation_count": count}
            for name, count in activity_summary.most_common()
        ]

        summary = {
            "student_count": len(students),
            "activity_count": len(activities),
            "total_participations": int(sum(item["participation_count"] for item in students)),
        }
        return {
            "summary": summary,
            "rankings": rankings,
            "students": students,
            "activities": activities,
        }

    def export_attendance_to_csv(self, date=None):
        rows = self.get_attendance_by_date(date) if date else self.get_today_attendance()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["学号", "姓名", "班级", "打卡时间", "检测方法", "识别置信度", "活体分数", "情绪", "来源", "活动名称"])
        for row in rows:
            writer.writerow(
                [
                    row["student_id"],
                    row["student_name"],
                    row["class_name"],
                    row["check_time"],
                    row["detection_method"],
                    row["confidence"],
                    row["liveness_score"],
                    row["emotion"],
                    row["source"],
                    row["activity_name"] or "无",
                ]
            )
        return output.getvalue()
