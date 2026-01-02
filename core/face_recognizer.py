from pathlib import Path
import numpy as np
import torch

from utils.face_utils import resnet, device


class FaceRecognizer:
    def __init__(self, db_path=None, threshold=0.6):
        if db_path is None:
            current_dir = Path(__file__).parent
            db_path = current_dir.parent / "embeddings" / "database.npz"

        try:
            data = np.load(db_path, allow_pickle=True)
            self.known_names = data['names']
            self.known_embeddings = data['embeddings']
            self.base_names = set(self.known_names)
            self.threshold = threshold
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ 没有找到数据库！请先运行 build_database.py\n"
                f"🔍 路径: {db_path.absolute()}"
            )

    @staticmethod
    def cosine_similarity(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def recognize(self, face_pil):
        with torch.no_grad():
            emb = resnet(face_pil.unsqueeze(0).to(device)).cpu().numpy().flatten()

        max_sim = -1
        best_match = "识别失败"
        for name, emb_db in zip(self.known_names, self.known_embeddings):
            sim = self.cosine_similarity(emb, emb_db)
            if sim > max_sim:
                max_sim = sim
                best_match = name
        return best_match if max_sim >= self.threshold else "识别失败"

    def get_all_registered_names(self):
        return self.base_names.copy()