import threading
from datetime import datetime

# 新增：记录已签到的学生
signed_in_students = set()
signin_lock = threading.Lock()

import cv2
import numpy as np
import torch
from utils.face_utils import mtcnn, resnet, device
from PIL import Image, ImageDraw, ImageFont

cv2.setUseOptimized(True)

# 在加载数据库的部分添加基础名单的加载
try:
    data = np.load('embeddings/database.npz', allow_pickle=True)
    known_names = data['names']
    known_embeddings = data['embeddings']
    print(f"已加载注册学生人脸信息。")

    # 加载基础名单（假定与数据库同名）
    base_names = set(data['names'])  # 使用集合便于后续操作
except FileNotFoundError:
    print("❌ 没有找到数据库！请先运行 build_database.py")
    exit()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(face_emb, threshold=0.6):
    max_sim = -1
    best_match = "Unknown"
    for name, emb in zip(known_names, known_embeddings):
        sim = cosine_similarity(face_emb, emb)
        if sim > max_sim:
            max_sim = sim
            best_match = name
    return best_match if max_sim >= threshold else "Unknown"


from PIL import Image

def get_chinese_font(size=32):
    """尝试加载系统中文字体"""
    font_paths = [
        "simhei.ttf",     # 黑体
        "msyh.ttc",       # 微软雅黑
        "simsun.ttc",     # 宋体
        "arialuni.ttf"    # 通用 Unicode 字体（部分系统有）
    ]
    for font in font_paths:
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            continue
    # 如果都没找到，返回默认字体（可能不支持中文，但不会崩溃）
    return ImageFont.load_default()

def show_signin_popup(name):
    """显示 1 秒签到成功提示（支持中文）"""
    # 创建 PIL 图像
    img = Image.new('RGB', (400, 200), color=(0, 0, 0))  # 黑底
    draw = ImageDraw.Draw(img)

    # 使用中文字体（推荐宋体/微软雅黑）
    try:
        # 尝试加载系统字体（Windows 通常有这些）
        font = ImageFont.truetype("simhei.ttf", 32)  # 黑体
    except:
        try:
            font = ImageFont.truetype("msyh.ttc", 32)  # 微软雅黑
        except:
            font = ImageFont.load_default()  # 备用字体（可能不支持中文）

    # 绘制中文文本
    text = f"{name}同学签到成功！"
    draw.text((50, 100), text, fill=(0, 255, 0), font=font)

    # 转换为 OpenCV 格式
    cv_img = np.array(img)
    cv_img = cv_img[:, :, ::-1]  # RGB → BGR

    # 显示弹窗
    cv2.imshow("签到成功", cv_img)
    cv2.waitKey(1000)
    cv2.destroyWindow("签到成功")

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("按 “q” 退出。")

# ===== 主循环：持续运行 =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_img = rgb_frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            face_pil = mtcnn(Image.fromarray(face_img))
            if face_pil is not None:
                with torch.no_grad():
                    emb = resnet(face_pil.unsqueeze(0).to(device)).cpu().numpy().flatten()
                name = recognize_face(emb)

                if name != "Unknown":
                    with signin_lock:
                        if name not in signed_in_students:
                            signed_in_students.add(name)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {name} 已签到")
                            threading.Thread(target=show_signin_popup, args=(name,), daemon=True).start()

                # === 替换这里：支持中文显示 ===
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if name != "Unknown":
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font = get_chinese_font(size=30)
                    text_position = (x1, max(y1 - 35, 0))
                    draw.text(text_position, name, fill=(0, 255, 0), font=font)
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Face Sign-in System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ===== 循环结束后才执行以下代码 =====
cap.release()
cv2.destroyAllWindows()

# 保存签到记录
if signed_in_students:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"signin_record_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("课堂人脸签到记录\n")
        f.write("=" * 30 + "\n")
        for student in sorted(signed_in_students):
            f.write(f"{student}\n")

        # 添加未签到的学生信息
        unsigned_students = base_names - signed_in_students
        if unsigned_students:
            f.write("\n未签到的学生:\n")
            f.write("=" * 30 + "\n")
            for student in sorted(unsigned_students):
                f.write(f"{student}\n")

    print(f"\n✅ 签到记录已保存至: {filename}")
else:
    print("\nℹ️ 无人签到。")