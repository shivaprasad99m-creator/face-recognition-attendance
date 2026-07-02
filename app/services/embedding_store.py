from __future__ import annotations

import numpy as np

from app.db.database import Database


def serialize_embedding(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def deserialize_embedding(blob: bytes, dimension: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(dimension)


class EmbeddingStore:
    def __init__(self, database: Database):
        self.database = database

    def upsert_user(self, external_id: str, name: str) -> dict:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO users (external_id, name)
                VALUES (?, ?)
                ON CONFLICT(external_id) DO UPDATE SET name = excluded.name
                """,
                (external_id, name),
            )
            row = connection.execute(
                "SELECT id, external_id, name, created_at FROM users WHERE external_id = ?",
                (external_id,),
            ).fetchone()
            return dict(row)

    def add_embedding(self, user_id: int, embedding: np.ndarray, model_name: str) -> int:
        vector = np.asarray(embedding, dtype=np.float32)
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO face_embeddings (user_id, embedding, dimension, model_name)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, serialize_embedding(vector), vector.size, model_name),
            )
            return int(cursor.lastrowid)

    def list_users(self) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id, external_id, name, created_at FROM users ORDER BY name"
            ).fetchall()
            return [dict(row) for row in rows]

    def load_embeddings(self) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    e.id AS embedding_id,
                    e.embedding,
                    e.dimension,
                    u.id AS user_id,
                    u.external_id,
                    u.name
                FROM face_embeddings e
                JOIN users u ON u.id = e.user_id
                """
            ).fetchall()

        embeddings = []
        for row in rows:
            item = dict(row)
            item["embedding"] = deserialize_embedding(item["embedding"], item["dimension"])
            embeddings.append(item)
        return embeddings
