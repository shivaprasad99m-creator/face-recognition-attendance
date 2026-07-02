from __future__ import annotations

from app.db.database import Database


class AttendanceService:
    def __init__(self, database: Database, duplicate_window_minutes: int):
        self.database = database
        self.duplicate_window_minutes = duplicate_window_minutes

    def mark(self, user_id: int, confidence: float, source: str = "api") -> tuple[bool, int | None]:
        with self.database.connect() as connection:
            recent = connection.execute(
                """
                SELECT id
                FROM attendance_logs
                WHERE user_id = ?
                  AND recognized_at >= datetime('now', ?)
                ORDER BY recognized_at DESC
                LIMIT 1
                """,
                (user_id, f"-{self.duplicate_window_minutes} minutes"),
            ).fetchone()
            if recent:
                return False, int(recent["id"])

            cursor = connection.execute(
                """
                INSERT INTO attendance_logs (user_id, confidence, source)
                VALUES (?, ?, ?)
                """,
                (user_id, confidence, source),
            )
            return True, int(cursor.lastrowid)

    def list_logs(self, limit: int = 100) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    a.id,
                    a.user_id,
                    u.external_id,
                    u.name,
                    a.recognized_at,
                    a.confidence,
                    a.source
                FROM attendance_logs a
                JOIN users u ON u.id = a.user_id
                ORDER BY a.recognized_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
