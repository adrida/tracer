import unittest
import numpy as np
import json
from pathlib import Path
import shutil
import os

from tracer.analysis.qualitative import build_qualitative_report
from tracer.embeddings.index import EmbeddingIndex
from tracer.fit.pipeline import _split_buffer
from tracer.api import update, fit
from tracer.config import FitConfig

class TestBugFixes(unittest.TestCase):

    def test_empty_report(self):
        """Test build_qualitative_report with empty inputs."""
        report = build_qualitative_report(texts=[], teacher_labels=[], decisions=[])
        self.assertEqual(report.summary, "No traces provided.")
        self.assertEqual(report.coverage, 0.0)

    def test_index_fallback_cosine(self):
        """Test EmbeddingIndex fallback for cosine without FAISS."""
        X = np.eye(3).astype(np.float32)
        # Manually create index with None to force fallback
        idx = EmbeddingIndex(X, index=None, metric="cosine")
        
        # Query exactly matches second vector
        query = np.array([0, 1, 0], dtype=np.float32)
        I, D = idx.search(query, k=1)
        
        self.assertEqual(I[0], 1)
        self.assertAlmostEqual(D[0], 1.0) # Inner product of [0,1,0] and [0,1,0]

    def test_small_split(self):
        """Test _split_buffer with n=2 per class."""
        X = np.random.randn(4, 10)
        y = np.array([0, 0, 1, 1])
        split = _split_buffer(X, y)
        
        # Previously val_idx was empty for n=2. Now it should have 1 per class.
        self.assertEqual(len(split["X_val"]), 2)
        self.assertEqual(len(split["X_train"]), 2)
        self.assertEqual(len(split["X_cal"]), 0)

    def test_update_dim_check(self):
        """Test update() raises ValueError on dimension mismatch."""
        tmp_dir = Path("test_tmp_update")
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        
        try:
            # Create a dummy fit
            traces_path = tmp_dir / "traces.jsonl"
            traces_path.write_text(json.dumps({"input": "hi", "teacher": "A", "id": "1"}) + "\n")
            X = np.random.randn(1, 10).astype(np.float32)
            
            # Using fit directly to setup artifacts
            fit(traces_path, artifact_dir=tmp_dir / ".tracer", embeddings=X)
            
            # Try to update with 5-dim embeddings instead of 10
            X_new = np.random.randn(1, 5).astype(np.float32)
            new_traces_path = tmp_dir / "new.jsonl"
            new_traces_path.write_text(json.dumps({"input": "bye", "teacher": "B", "id": "2"}) + "\n")
            
            with self.assertRaises(ValueError) as cm:
                update(new_traces_path, artifact_dir=tmp_dir / ".tracer", new_embeddings=X_new)
            
            self.assertIn("Embedding dimension mismatch", str(cm.exception))
        finally:
            shutil.rmtree(tmp_dir)

    def test_index_persistence(self):
        """Test EmbeddingIndex save/load preserves the metric."""
        tmp_path = Path("test_index_metric")
        X = np.eye(3).astype(np.float32)
        idx = EmbeddingIndex(X, metric="cosine")
        idx.save(tmp_path)
        
        try:
            loaded = EmbeddingIndex.load(tmp_path)
            self.assertEqual(loaded.metric, "cosine")
        finally:
            for p in [".embeddings.npy", ".metric", ".faiss"]:
                if tmp_path.with_suffix(p).exists():
                    tmp_path.with_suffix(p).unlink()

if __name__ == "__main__":
    unittest.main()
