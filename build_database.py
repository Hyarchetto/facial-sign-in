import os
import numpy as np
from utils.face_utils import extract_embedding

DATASET_DIR = 'dataset'
EMBEDDINGS_DIR = 'embeddings'

# 确保 embeddings 目录存在（关键：这行已经足够！）
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def build_database():
    names = []
    embeddings = []

    # 检查 dataset 目录是否存在
    if not os.path.exists(DATASET_DIR):
        print(f"❌ 数据集目录 '{DATASET_DIR}' 不存在，请先创建并放入人脸图片！")
        return

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"处理 {person_name} 中 ...")
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(person_dir, img_name)
            emb = extract_embedding(img_path)
            if emb is not None:
                names.append(person_name)
                embeddings.append(emb)
                print(f"  -> 从 {img_name} 中提取")
            else:
                print(f"  -> 从 {img_name} 中没有检测到人脸")

    if embeddings:
        save_path = os.path.join(EMBEDDINGS_DIR, 'database.npz')
        np.savez(save_path, names=np.array(names), embeddings=np.array(embeddings))
        print(f"\n✅ 数据库保存至: {save_path}")
        print(f"   已录入: {len(embeddings)} 张有效图片，共 {len(set(names))} 人")
    else:
        print("❌ 没有找到有效的面孔。检查你的数据集！")

if __name__ == '__main__':
    build_database()