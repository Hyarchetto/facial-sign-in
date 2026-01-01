import os
import threading
import time
from datetime import datetime
import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model
from utils.face_utils import mtcnn, resnet, device
from PIL import Image, ImageDraw, ImageFont

# ======================
# 全局配置与初始化
# ======================

signed_in_students = set()
signin_lock = threading.Lock()

def get_chinese_font(size=30):
    font_paths = [
        "simhei.ttf",
        "msyh.ttc",
        "simsun.ttc",
        "arialuni.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()

CHINESE_FONT = get_chinese_font(30)

# 加载人脸数据库
try:
    data = np.load('embeddings/database.npz', allow_pickle=True)
    known_names = data['names']
    known_embeddings = data['embeddings']
    base_names = set(known_names)
except FileNotFoundError:
    print("❌ 没有找到数据库！请先运行 build_database.py")
    exit()

# 加载口罩检测模型
MASK_MODEL_PATH = 'mask_model/mask_detector.model'
try:
    mask_model = load_model(MASK_MODEL_PATH)
except Exception as e:
    print(f"❌ 无法加载口罩检测模型: {e}")
    exit()

def detect_mask(face_img):
    """判断是否戴口罩，True 表示戴口罩"""
    try:
        resized = cv2.resize(face_img, (224, 224))
        normalized = resized.astype("float") / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        preds = mask_model.predict(input_tensor, verbose=0)  # 关闭 Keras 日志
        return preds[0][0] > 0.99  # 提高阈值以减少误判
    except:
        return False

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def recognize_face(face_emb, threshold=0.6):
    max_sim = -1
    best_match = "识别失败"
    for name, emb in zip(known_names, known_embeddings):
        sim = cosine_similarity(face_emb, emb)
        if sim > max_sim:
            max_sim = sim
            best_match = name
    return best_match if max_sim >= threshold else "识别失败"

# ======================
# 主程序
# ======================

def main():
    global signed_in_students

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("按 “q” 退出签到系统。")

    current_faces = []
    last_signin_msg = ""
    last_signin_time = 0.0
    frame_count = 0
    process_every_n_frames = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)

            new_faces = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    if detect_mask(face_crop):
                        new_faces.append((x1, y1, x2, y2, "请取下口罩"))
                        continue

                    face_pil = mtcnn(Image.fromarray(face_crop))
                    if face_pil is None:
                        continue

                    with torch.no_grad():
                        emb = resnet(face_pil.unsqueeze(0).to(device)).cpu().numpy().flatten()
                    name = recognize_face(emb)

                    if name != "识别失败":
                        with signin_lock:
                            if name not in signed_in_students:
                                signed_in_students.add(name)
                                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                print(f"[{current_time_str}] ✅ {name} 同学签到成功！")
                                last_signin_msg = f"{name}同学签到成功！"
                                last_signin_time = time.time()

                    new_faces.append((x1, y1, x2, y2, name))

            current_faces = new_faces

        # 渲染画面
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for (x1, y1, x2, y2, name) in current_faces:

            if name == "识别失败":
                color = (255, 0, 0)      # 红色
            elif name == "请取下口罩":
                color = (255, 255, 0)    # 黄色
            else:
                color = (0, 255, 0)      # 绿色

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text = name + ("(已签到)" if name != "识别失败" and name != "请取下口罩" else "")
            draw.text((x1, max(y1 - 35, 0)), text, fill=color, font=CHINESE_FONT)

        # 显示短暂的签到成功提示
        if time.time() - last_signin_time < 1.0 and last_signin_msg:
            draw.text((20, 20), last_signin_msg, fill=(0, 255, 0), font=CHINESE_FONT)

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Sign-in System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 保存签到记录
    if signed_in_students:
        record_dir = "Signin-record"
        os.makedirs(record_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(record_dir, f"signin_record_{timestamp}.txt")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("课堂人脸签到记录\n")
            f.write("=" * 30 + "\n")
            for student in sorted(signed_in_students):
                f.write(f"{student}\n")
            unsigned = base_names - signed_in_students
            if unsigned:
                f.write("\n未签到的学生:\n")
                f.write("=" * 30 + "\n")
                for student in sorted(unsigned):
                    f.write(f"{student}\n")

        print(f"\n✅ 签到记录已保存至: {filename}")
    else:
        print("\nℹ️ 无人签到。")

if __name__ == '__main__':
    main()