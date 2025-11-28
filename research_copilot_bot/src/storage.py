import sqlite_utils
import faiss
import numpy as np
import json
import os
import threading

class Storage:
    def __init__(self, db_path: str, vector_store_type: str = "faiss"):
        self.db_path = db_path
        self.vector_store_type = vector_store_type
        self.index = None
        self.chunks = [] # Keep chunks in memory for simple retrieval mapping
        self.lock = threading.Lock()
        self._init_db()

    def _get_db(self):
        # Create a new connection for each operation to ensure thread safety
        return sqlite_utils.Database(self.db_path)

    def _init_db(self):
        db = self._get_db()
        db["papers"].create({
            "id": str,
            "path": str,
            "meta": str # JSON
        }, pk="id", if_not_exists=True)

        db["chunks"].create({
            "id": int, # index in FAISS
            "paper_id": str,
            "text": str,
            "start": int,
            "end": int
        }, pk="id", if_not_exists=True)

        db["summaries"].create({
            "paper_id": str,
            "summary": str # JSON
        }, pk="paper_id", if_not_exists=True)

    def add_paper(self, paper_id: str, path: str, meta: dict):
        db = self._get_db()
        db["papers"].insert({
            "id": paper_id,
            "path": path,
            "meta": json.dumps(meta)
        }, replace=True)

    def add_chunks(self, paper_id: str, chunks: list, embeddings: list):
        # Add to FAISS (Thread-safe with lock)
        with self.lock:
            if self.index is None:
                dim = len(embeddings[0])
                self.index = faiss.IndexFlatL2(dim)

            start_idx = self.index.ntotal
            self.index.add(np.array(embeddings).astype('float32'))

        # Add to DB
        db = self._get_db()
        rows = []
        for i, chunk in enumerate(chunks):
            rows.append({
                "id": start_idx + i,
                "paper_id": paper_id,
                "text": chunk["text"],
                "start": chunk["start"],
                "end": chunk["end"]
            })
        db["chunks"].insert_all(rows, pk="id", replace=True)

    def get_summary(self, paper_id: str):
        db = self._get_db()
        try:
            row = db["summaries"].get(paper_id)
            return json.loads(row["summary"]) if row else None
        except sqlite_utils.db.NotFoundError:
            return None

    def save_summary(self, paper_id: str, summary: dict):
        db = self._get_db()
        db["summaries"].insert({
            "paper_id": paper_id,
            "summary": json.dumps(summary)
        }, replace=True)

    def retrieve(self, query_embedding: list, k: int = 5):
        # Search is generally thread-safe in FAISS for FlatL2, but locking doesn't hurt
        with self.lock:
            if self.index is None:
                return []
            D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)

        indices = I[0]
        results = []
        db = self._get_db()
        for idx in indices:
            if idx == -1: continue
            try:
                row = db["chunks"].get(int(idx))
                if row:
                    results.append(row)
            except sqlite_utils.db.NotFoundError:
                pass
        return results
