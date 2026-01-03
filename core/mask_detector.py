import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path


class MaskDetector:
    def __init__(self, model_path=None):
        """
        初始化口罩检测器

        Args:
            model_path (str): 模型路径，若为 None 则自动查找项目根目录下的 mask_model/mask_detector.model
        """
        if model_path is None:
            # 获取当前文件所在目录的上两级的项目根目录
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "mask_model" / "mask_detector.model"

        try:
            self.model = load_model(model_path)
            print(f"✅ 口罩检测模型已加载: {model_path.absolute()}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ 无法找到口罩检测模型！\n"
                f"🔍 尝试路径: {model_path.absolute()}\n"
                f"💡 请确保模型文件在 'mask_model/' 目录中，并命名为 'mask_detector.model'"
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ 加载口罩检测模型失败: {e}\n"
                f"🔍 模型路径: {model_path.absolute()}"
            )

    def detect(self, face_rgb):
        """
        检测是否戴口罩

        Args:
            face_rgb (np.ndarray): RGB 图像数组 (H, W, C)

        Returns:
            bool: True 表示戴口罩，False 表示未戴
        """
        try:
            resized = cv2.resize(face_rgb, (224, 224))
            normalized = resized.astype("float") / 255.0
            input_tensor = np.expand_dims(normalized, axis=0)
            preds = self.model.predict(input_tensor, verbose=0)
            return preds[0][0] > 0.99  # 高阈值减少误判
        except Exception as e:
            print(f"⚠️ 口罩检测失败: {e}")
            return False  # 默认认为未戴口罩