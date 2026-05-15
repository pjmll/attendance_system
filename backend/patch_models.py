import sqlite3

def add_activity_record(self, activity_name, student_id, student_name):
    '''添加活动合照考勤记录'''
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO activity_records (activity_name, student_id, student_name)
            VALUES (?, ?, ?)
        ''', (activity_name, student_id, student_name))
        conn.commit()
        return True
    except Exception as e:
        print(f'添加活动记录失败: {e}')
        return False
    finally:
        conn.close()

def get_activity_stats(self):
    '''获取各学生活动参与次数统计'''
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT 
                student_id, 
                student_name, 
                COUNT(*) as activity_count
            FROM activity_records
            GROUP BY student_id, student_name
            ORDER BY activity_count DESC
        ''')
        results = []
        for row in cursor.fetchall():
            results.append({
                'student_id': row[0],
                'student_name': row[1],
                'activity_count': row[2]
            })
        return results
    except Exception as e:
        print(f'获取活动统计失败: {e}')
        return []
    finally:
        conn.close()
