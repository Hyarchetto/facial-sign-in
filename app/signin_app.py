import threading
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw
from core.face_recognizer import FaceRecognizer
from core.mask_detector import MaskDetector
from utils.font_utils import get_chinese_font
from recorder.record_saver import save_signin_record

# 全局状态
signed_in_students = set()
signin_lock = threading.Lock()
CHINESE_FONT = get_chinese_font(30)


def main():
    global signed_in_students

    # 初始化核心组件
    try:
        recognizer = FaceRecognizer()
        mask_detector = MaskDetector()
        base_names = recognizer.get_all_registered_names()
    except Exception as e:
        print(e)
        return

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
            from utils.face_utils import mtcnn  # 延迟导入避免循环
            boxes, _ = mtcnn.detect(rgb_frame)

            new_faces = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # 口罩检测
                    if mask_detector.detect(face_crop):
                        new_faces.append((x1, y1, x2, y2, "请取下口罩"))
                        continue

                    # 人脸识别
                    face_pil = mtcnn(Image.fromarray(face_crop))
                    if face_pil is None:
                        new_faces.append((x1, y1, x2, y2, "识别失败"))
                        continue

                    name = recognizer.recognize(face_pil)
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

        # 渲染
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for (x1, y1, x2, y2, name) in current_faces:
            color = (255, 0, 0)  # 默认红
            if name == "请取下口罩":
                color = (255, 255, 0)  # 黄
            elif name != "识别失败":
                color = (0, 255, 0)  # 绿

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text = name + ("(已签到)" if name not in ["识别失败", "请取下口罩"] else "")
            draw.text((x1, max(y1 - 35, 0)), text, fill=color, font=CHINESE_FONT)

        if time.time() - last_signin_time < 1.0 and last_signin_msg:
            draw.text((20, 20), last_signin_msg, fill=(0, 255, 0), font=CHINESE_FONT)

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Sign-in System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_signin_record(signed_in_students, base_names)


if __name__ == '__main__':
    main()