import argparse
import logging
import os
import sys

from ocr_namer import MAX_STEPS, ocr_namer
from process import log, total_process_handler

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
    parser.add_argument("image_path", nargs="?", default=None,
                        help="照片目录路径（支持递归子目录），不指定则使用 config_local.py 中的默认路径")
    parser.add_argument("--skip-named", action="store_true",
                        help="跳过已命名文件，不重新验证（默认会重新 OCR 验证）")
    args = parser.parse_args()

    # 未指定路径时使用 config_local.py 中的默认路径
    if args.image_path is not None:
        image_path = args.image_path
    else:
        try:
            from config_local import DEFAULT_IMAGE_PATH
            image_path = DEFAULT_IMAGE_PATH
        except ImportError:
            log("ERROR", "未指定目录路径，且 config_local.py 不存在")
            log("INFO", "请通过命令行参数指定路径，或创建 config_local.py 设置 DEFAULT_IMAGE_PATH")
            sys.exit(1)
    if not os.path.isdir(image_path):
        log("ERROR", "目录不存在: {}".format(image_path))
        sys.exit(1)

    images = collect_images(image_path)
    log("INFO", "共找到 {} 张图片 | 目录: {}".format(len(images), image_path))
    total_process_handler.steps = len(images) * MAX_STEPS
    for dir_path, file_name in images:
        try:
            ocr_namer(dir_path, file_name, skip_named=args.skip_named)
        except Exception as e:
            log("ERROR", "处理出错，跳过: {}".format(e), file_name)


if __name__ == "__main__":
    main()
