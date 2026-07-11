import os
import os.path
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFilter

from get_name_number import GetNameNumber
from process import ProcessHandler, total_process_handler
from regex import bpt_line_regex, non_bpt_line_regex

# 最大 OCR 轮次：Phase1(3) + Phase2(9) + Phase3(9) = 21
MAX_STEPS = 21


def otsu_threshold(image_array: np.ndarray) -> int:
    """使用 Otsu 方法计算最佳二值化阈值。"""
    hist, _ = np.histogram(image_array, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(np.arange(256) * hist)
    weight_bg = cum_sum
    weight_fg = total - cum_sum
    mean_bg = np.divide(cum_mean, np.maximum(weight_bg, 1))
    mean_fg = np.divide(cum_mean[-1] - cum_mean, np.maximum(weight_fg, 1))
    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between_var))


def has_enough_results(line_list: list, number_list: list, id_list: list) -> bool:
    """检查是否已有足够的检测结果用于置信度投票（每个字段至少 2 个）。"""
    return len(line_list) >= 2 and len(number_list) >= 2 and len(id_list) >= 2


def run_ocr_batch(file_name: str, images: list, process_handler: ProcessHandler) -> tuple:
    """并行执行一批图像变体的 OCR，返回合并后的 (line_list, number_list, id_list)。"""
    threads = []
    for im in images:
        thread = GetNameNumber(file_name, im, process_handler)
        thread.start()
        threads.append(thread)
    line_list = []
    number_list = []
    id_list = []
    for thread in threads:
        thread.join()
        result = thread.get_result()
        if result is not None:
            l, n, i = result
            line_list += l
            number_list += n
            id_list += i
    return line_list, number_list, id_list


def ocr_namer(file_path: str, file_name: str):
    """对单张公交车照片执行 OCR 识别并重命名。

    采用三阶段递进策略，在检测结果充足时提前退出，避免无谓的 OCR 调用：
    Phase 1: 3 张基础图像（原图 / 模糊锐化 / 缩放锐化）
    Phase 2: 9 张 RGB 通道拆分（仅在 Phase 1 不足时）
    Phase 3: 9 张 Otsu 自适应二值化（仅在 Phase 2 不足时）
    每阶段内的图像变体通过多线程并行处理。
    """
    # 跳过已识别格式的图片（文件名以"{线路号}路"开头）
    line_prefix_match = re.match(r'^(.+?)路', file_name)
    if line_prefix_match:
        prefix = line_prefix_match.group(1)
        if re.fullmatch(bpt_line_regex, prefix) or re.fullmatch(non_bpt_line_regex, prefix):
            print("[{}]跳过已识别格式的图片".format(file_name))
            return
    single_process_handler = ProcessHandler(MAX_STEPS)
    print("[{} {:.2f}% {:.2f}%]processing {}".format(file_name,
                                                     single_process_handler.process,
                                                     total_process_handler.process,
                                                     file_name))
    line_list = []
    number_list = []
    id_list = []
    try:
        image = Image.open(os.path.join(file_path, file_name)).convert("RGB")
    except Exception as e:
        print("[{} {:.2f}% {:.2f}%]图片加载失败，跳过：{}".format(file_name,
                                                             single_process_handler.process,
                                                             total_process_handler.process, e))
        return
    base_images = [
        image,
        image.filter(ImageFilter.GaussianBlur(radius=2)).filter(ImageFilter.EDGE_ENHANCE),
        image.resize((1280, 720)).filter(ImageFilter.GaussianBlur(radius=2)).filter(ImageFilter.EDGE_ENHANCE),
    ]

    # Phase 1：基础图像（多线程并行）
    l, n, i = run_ocr_batch(file_name, base_images, single_process_handler)
    line_list += l
    number_list += n
    id_list += i
    print("[{} {:.2f}% {:.2f}%]Phase 1 完成：线路号 {} 个，自编号 {} 个，车牌号 {} 个".format(
        file_name, single_process_handler.process, total_process_handler.process,
        len(line_list), len(number_list), len(id_list)))

    # Phase 2：RGB 通道拆分（仅在 Phase 1 结果不足时执行）
    if not has_enough_results(line_list, number_list, id_list):
        channel_images = []
        for im in base_images:
            r, g, b = im.split()
            channel_images.extend([r, g, b])
        l, n, i = run_ocr_batch(file_name, channel_images, single_process_handler)
        line_list += l
        number_list += n
        id_list += i
        print("[{} {:.2f}% {:.2f}%]Phase 2 完成：线路号 {} 个，自编号 {} 个，车牌号 {} 个".format(
            file_name, single_process_handler.process, total_process_handler.process,
            len(line_list), len(number_list), len(id_list)))

    # Phase 3：Otsu 自适应二值化（仅在 Phase 2 结果不足时执行）
    if not has_enough_results(line_list, number_list, id_list):
        binary_images = []
        for im in base_images:
            r, g, b = im.split()
            for image_mono in [r, g, b]:
                threshold = otsu_threshold(np.array(image_mono))
                binary_images.append(binary_image(image_mono, threshold))
        l, n, i = run_ocr_batch(file_name, binary_images, single_process_handler)
        line_list += l
        number_list += n
        id_list += i
        print("[{} {:.2f}% {:.2f}%]Phase 3 完成：线路号 {} 个，自编号 {} 个，车牌号 {} 个".format(
            file_name, single_process_handler.process, total_process_handler.process,
            len(line_list), len(number_list), len(id_list)))

    print("[{} {:.2f}% {:.2f}%]疑似线路号：{}\n疑似自编号：{}\n疑似车牌号：{}".format(
        file_name,
        single_process_handler.process,
        total_process_handler.process,
        [x[0] for x in line_list], [x[0] for x in number_list], [x[0] for x in id_list]))
    # 确定性去重：当线路号是其他更长候选值的子串且出现次数不超过容器时，删除
    line_counts = Counter(l[0] for l in line_list)
    number_counts = Counter(n[0] for n in number_list)
    id_counts = Counter(i[0] for i in id_list)
    # 合并所有"容器"候选（自编号、车牌号）及其出现次数
    containers = {}
    for text, count in number_counts.items():
        containers[text] = containers.get(text, 0) + count
    for text, count in id_counts.items():
        containers[text] = containers.get(text, 0) + count
    new_line_list = []
    for line_text, line_score in line_list:
        delete_flag = False
        container_text = ""
        container_count = 0
        # 检查是否是某个自编号/车牌号的子串
        for c_text, c_count in containers.items():
            if line_text in c_text and line_text != c_text:
                if line_counts[line_text] <= c_count:
                    delete_flag = True
                    container_text = c_text
                    container_count = c_count
                break
        # 检查是否是某个更长线路号的子串
        if delete_flag is False:
            for line2_text, line2_count in line_counts.items():
                if len(line_text) < len(line2_text) and line_text in line2_text:
                    if line_counts[line_text] <= line2_count:
                        delete_flag = True
                        container_text = line2_text
                        container_count = line2_count
                    break
        if delete_flag is True:
            print("[{} {:.2f}% {:.2f}%]清理：疑似线路号 {} 包含在 {} 中（出现 {} 次 ≤ {} 次）".format(
                file_name,
                single_process_handler.process,
                total_process_handler.process, line_text, container_text,
                line_counts[line_text], container_count))
        else:
            new_line_list.append((line_text, line_score))
    line_list = new_line_list
    print("[{} {:.2f}% {:.2f}%]清理后：\n疑似线路号：{}\n疑似自编号：{}\n疑似车牌号：{}".format(
        file_name,
        single_process_handler.process,
        total_process_handler.process,
        [x[0] for x in line_list], [x[0] for x in number_list],
        [x[0] for x in id_list]))
    # 置信度加权投票：累计每个候选值的置信度，取最高者
    flag = False
    if len(line_list) > 0:
        weighted = defaultdict(float)
        for text, score in line_list:
            weighted[text] += score
        line = max(weighted, key=weighted.get)
    else:
        line = "unknown"
        flag = True
    if len(number_list) > 0:
        weighted = defaultdict(float)
        for text, score in number_list:
            weighted[text] += score
        number = max(weighted, key=weighted.get)
    else:
        number = "unknown"
        flag = True
    if len(id_list) > 0:
        weighted = defaultdict(float)
        for text, score in id_list:
            weighted[text] += score
        id_ = max(weighted, key=weighted.get).replace("皖", "京")
    else:
        id_ = "unknown"
        flag = True
    # 所有字段均为 unknown 时跳过重命名
    if line == "unknown" and number == "unknown" and id_ == "unknown":
        print("[{} {:.2f}% {:.2f}%]所有字段均为 unknown，跳过重命名".format(
            file_name, single_process_handler.process, total_process_handler.process))
        return
    if re.match(non_bpt_line_regex, line) is not None:
        # 非公交集团线路用车牌号
        if flag is True:
            new_file_name = "{}路{}_{}.jpg".format(line, id_, file_name.split(".")[0])
        else:
            new_file_name = "{}路{}.jpg".format(line, id_)
    else:
        if flag is True:
            new_file_name = "{}路{}_{}_{}.jpg".format(line, number, id_, file_name.split(".")[0])
        else:
            new_file_name = "{}路{}_{}.jpg".format(line, number, id_)
    try:
        os.rename(os.path.join(file_path, file_name), os.path.join(file_path, new_file_name))
    except FileExistsError:
        new_file_name = "{}路{}_{}_{}.jpg".format(line, number, id_, file_name.split(".")[0])
        os.rename(os.path.join(file_path, file_name), os.path.join(file_path, new_file_name))
    print("[{} {:.2f}% {:.2f}%]{} -> {}".format(file_name,
                                                single_process_handler.process,
                                                total_process_handler.process,
                                                file_name,
                                                new_file_name))


def binary_image(image: Image, threshold: int = 128):
    """对图像进行二值化处理。"""
    table = [0 if i < threshold else 1 for i in range(256)]
    return image.point(table, "1")
