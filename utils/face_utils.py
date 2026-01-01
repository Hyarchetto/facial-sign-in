import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cpu'  # 如果你有 GPU 且装了 CUDA，可改为 'cuda'

# 初始化模型
mtcnn = MTCNN(
    image_size=160,
    margin=10,
    min_face_size=20,
    thresholds=[0.7, 0.8, 0.8],
    factor=0.709,
    post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_embedding(image_path):
    """
    从单张图片中提取 128 维人脸特征向量
    :param image_path: 图片路径
    :return: embedding (numpy array) 或 None（如果没检测到人脸）
    """
    try:
        img = Image.open(image_path).convert('RGB')
        # 检测并裁剪人脸
        img_cropped = mtcnn(img)
        if img_cropped is None:
            return None
        # 提取特征
        with torch.no_grad():
            embedding = resnet(img_cropped.unsqueeze(0).to(device))
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None