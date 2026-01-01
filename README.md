# 人脸签到系统（Face Sign-in System）

基于深度学习的人脸识别课堂签到工具，支持中文显示、防重复签到、口罩检测，并自动生成签到记录与未签到名单。

---

## 🌟 核心功能

👤 **人脸识别签到**  
- 自动识别已注册学生并记录签到状态  
- 同一学生仅允许签到一次（防刷签）

😷 **口罩检测拦截**  
- 若检测到佩戴口罩，提示“请取下口罩”  
- 不允许戴口罩状态下签到（提升识别准确率）

🖨️ **实时签到日志**  
- 首次成功签到时，**控制台自动打印带时间戳的日志**，例如：  
  `[2026-01-01 21:15:32] ✅ 张三 同学签到成功！`

📋 **签到结果导出**  
程序退出后自动生成文本文件，包含：
- ✅ 已签到学生名单  
- ❌ 未签到学生名单（对比 `dataset/` 中所有注册学生）

⚙️ **模块化设计**  
主程序、数据库构建、工具函数分离，便于维护与扩展

---
## 📂 项目结构
```
face-signin/
├── dataset/                # 【需手动准备】原始人脸图像（按姓名建子文件夹）
├── embeddings/             # 自动生成的人脸特征数据库
│ └── database.npz
├── mask_model/             # 口罩检测模型
│ └── mask_detector.model
├── utils/
│ └── face_utils.py         # MTCNN + ResNet 模型加载与人脸嵌入提取
├── build_database.py       # 构建人脸特征数据库
├── signin_app.py           # 主签到程序（含日志与口罩检测）
├── requirements.txt        # Python 依赖包列表
├── .gitignore
└── README.md               # 本说明文件
```

---

## 🚀 快速开始

 ### 1. 安装依赖

```bash
pip install -r requirements.txt
```

 ### 2. 准备人脸数据集

在项目根目录创建 dataset/ 文件夹，按学生姓名建立子文件夹，每个子文件夹放入该学生 3~5 张清晰正面照：
```
dataset/
├── 张三/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
├── 李四/
│   ├── img1.png
│   └── img2.png
└── 王五/
    └── photo.jpg
```

要求：
- 图像格式：`.jpg`, `.jpeg`, `.png`
- 正脸、无遮挡、光照均匀

 ### 3. 构建人脸特征数据库

```bash
python build_database.py
```

成功后将自动生成 embeddings/database.npz 文件。


⚠️ 若提示“找不到 dataset/”，请检查路径是否正确

 ### 4. 启动签到系统

```bash
python signin_app.py
```

- 摄像头画面实时检测人脸
- 识别成功且未签到 → 弹出绿色提示：“张三同学签到成功！”（1秒）
- 同时在控制台打印带时间的日志
- 未知人脸 → 显示红色“识别失败”
- 戴口罩 → 显示黄色“请取下口罩”
- 按键盘 q 键退出程序

 ### 5. 查看签到结果

程序退出后，自动生成带时间戳的签到记录文件，例如：

`Signin-record/signin_record_20260101_211532.txt`

内容示例：
```
课堂人脸签到记录
==============================
李四
张三

未签到的学生:
==============================
王五
赵六
```

---

## ⚙️ 技术依赖

详见 [`requirements.txt`](requirements.txt)

---

## ⚠️ 注意事项

- 首次运行前必须：
  - 准备好 `dataset/`
  - 运行 `build_database.py`
- 若提示 “❌ 没有找到数据库！...”，说明 `embeddings/database.npz` 不存在
- 中文字体依赖系统字体（Windows 默认支持 `simhei.ttf` / `msyh.ttc`）
- 本项目不上传人脸图像和特征数据库，符合隐私保护原则
- 如需支持更多中文系统，请在 get_chinese_font() 中扩展字体路径

---

## 📦 模型来源说明

本项目使用的口罩检测模型 `mask_model/mask_detector.model` 来源于开源项目  
[Face-Mask-Detection by Chandrika Deb](https://github.com/chandrikadeb7/Face-Mask-Detection)，  
遵循 **MIT License**。

> Copyright (c) 2020 Chandrika Deb  
> Permission is hereby granted, free of charge, to any person obtaining a copy  
> of this software and associated documentation files...

本仓库已包含该模型文件，可直接使用。
原始项目使用 MobileNetV2 架构训练，基于真实戴口罩人脸图像，准确率约 98%。  
感谢作者的贡献！

---

## 📅 最后更新
- 2026年01月