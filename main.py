import argparse
import logging
import os
import sys

from ocr_namer import MAX_STEPS, ocr_namer
from process import total_process_handler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.disable(logging.DEBUG)

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_images(root_path: str) -> list:
    """递归收集目录下所有图片文件，返回 (目录路径, 文件名) 列表。"""
    images = []
    for dir_path, _, file_names in os.walk(root_path):
        for file_name in file_names:
            if os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS:
                images.append((dir_path, file_name))
    return images


def main():
    """主入口：递归遍历指定目录及其子目录下的公交车照片，逐张执行 OCR 识别并重命名。"""
    parser = argparse.ArgumentParser(description="公交车照片 OCR 自动重命名工具")
    parser.add_argument("image_path", help="照片目录路径（支持递归子目录）")
    args = parser.parse_args()

    image_path = args.image_path
    if not os.path.isdir(image_path):
        print("目录不存在：{}".format(image_path))
        sys.exit(1)

    images = collect_images(image_path)
    print("共找到 {} 张图片".format(len(images)))
    total_process_handler.steps = len(images) * MAX_STEPS
    for dir_path, file_name in images:
        try:
            ocr_namer(dir_path, file_name)
        except Exception as e:
            print("处理 {} 时出错，跳过：{}".format(file_name, e))


if __name__ == "__main__":
    main()
