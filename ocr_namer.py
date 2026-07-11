import os
import os.path
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFilter

from get_name_number import GetNameNumber
from process import ProcessHandler, log, total_process_handler
from regex import bpt_line_regex, id_regex, non_bpt_line_regex

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


def parse_filename(file_name: str) -> tuple:
    """从已命名的文件名中提取线路号、自编号、车牌号。

    返回 (line, number, id_)，未找到的字段为 None。
    """
    stem = file_name.rsplit('.', 1)[0]
    line_match = re.match(r'^(.+?)路(.+)', stem)
    if not line_match:
        return None, None, None
    line = line_match.group(1)
    rest = line_match.group(2)
    # 提取车牌号
    id_match = re.search(id_regex, rest)
    id_ = id_match.group(0) if id_match else None
    # 提取自编号（路牌和车牌之间的部分，按 _ 分割取第一段）
    number = None
    if id_match:
        before_id = rest[:id_match.start()].strip('_')
        if before_id:
            number = before_id.split('_')[0]
    else:
        parts = rest.split('_')
        if parts and parts[0]:
            number = parts[0]
    return line, number, id_


def ocr_namer(file_path: str, file_name: str, skip_named: bool = False):
    """对单张公交车照片执行 OCR 识别并重命名。

    采用三阶段递进策略，在检测结果充足时提前退出，避免无谓的 OCR 调用：
    Phase 1: 3 张基础图像（原图 / 模糊锐化 / 缩放锐化）
    Phase 2: 9 张 RGB 通道拆分（仅在 Phase 1 不足时）
    Phase 3: 9 张 Otsu 自适应二值化（仅在 Phase 2 不足时）
    每阶段内的图像变体通过多线程并行处理。

    对已命名文件：重新 OCR 验证，如果结果与文件名不一致则用新结果替换。
    """
    # 判断是否为已命名文件（需要验证模式）
    verify_mode = False
    line_prefix_match = re.match(r'^(.+?)路', file_name)
    if line_prefix_match:
        prefix = line_prefix_match.group(1)
        if re.fullmatch(bpt_line_regex, prefix) or re.fullmatch(non_bpt_line_regex, prefix):
            if "unknown" not in file_name and len(prefix) > 1:
                if skip_named:
                    log("INFO", "跳过已命名文件", file_name)
                    return
                verify_mode = True
            else:
                log("INFO", "重新识别（unknown 或线路号为 1 位数）", file_name)
    single_process_handler = ProcessHandler(MAX_STEPS)
    sp = single_process_handler.process
    tp = total_process_handler.process
    if verify_mode:
        old_line, old_number, old_id = parse_filename(file_name)
        log("INFO", "验证模式 | 旧: 线路={}, 自编={}, 车牌={}".format(old_line, old_number, old_id),
            file_name, sp, tp)
    else:
        log("INFO", "开始处理 | 目录: {}".format(file_path), file_name, sp, tp)
    line_list = []
    number_list = []
    id_list = []
    try:
        image = Image.open(os.path.join(file_path, file_name)).convert("RGB")
    except Exception as e:
        log("ERROR", "图片加载失败，跳过: {}".format(e), file_name, sp, tp)
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
    sp = single_process_handler.process
    tp = total_process_handler.process
    log("INFO", "Phase 1 完成 | 线路号 {} 个, 自编号 {} 个, 车牌号 {} 个".format(
        len(line_list), len(number_list), len(id_list)), file_name, sp, tp)

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
        sp = single_process_handler.process
        tp = total_process_handler.process
        log("INFO", "Phase 2 完成 | 线路号 {} 个, 自编号 {} 个, 车牌号 {} 个".format(
            len(line_list), len(number_list), len(id_list)), file_name, sp, tp)

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
        sp = single_process_handler.process
        tp = total_process_handler.process
        log("INFO", "Phase 3 完成 | 线路号 {} 个, 自编号 {} 个, 车牌号 {} 个".format(
            len(line_list), len(number_list), len(id_list)), file_name, sp, tp)

    sp = single_process_handler.process
    tp = total_process_handler.process
    log("INFO", "疑似: 线路号={}, 自编号={}, 车牌号={}".format(
        [x[0] for x in line_list], [x[0] for x in number_list], [x[0] for x in id_list]),
        file_name, sp, tp)
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
            log("INFO", "清理: 线路号 {} 包含在 {} 中 ({} 次 ≤ {} 次)".format(
                line_text, container_text, line_counts[line_text], container_count),
                file_name, sp, tp)
        else:
            new_line_list.append((line_text, line_score))
    line_list = new_line_list
    log("INFO", "清理后: 线路号={}, 自编号={}, 车牌号={}".format(
        [x[0] for x in line_list], [x[0] for x in number_list], [x[0] for x in id_list]),
        file_name, sp, tp)
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
        # 优先选择京A开头的车牌号（北京公交集团车牌）
        jing_a_list = [(t, s) for t, s in id_list if t.startswith("京A")]
        if jing_a_list:
            weighted = defaultdict(float)
            for text, score in jing_a_list:
                weighted[text] += score
            id_ = max(weighted, key=weighted.get).replace("皖", "京")
            log("INFO", "优先选择京A车牌: {}".format(id_), file_name, sp, tp)
        else:
            weighted = defaultdict(float)
            for text, score in id_list:
                weighted[text] += score
            id_ = max(weighted, key=weighted.get).replace("皖", "京")
    else:
        id_ = "unknown"
        flag = True
    # 运通线路才允许 4 位自编号
    if number != "unknown" and len(number) == 4 and not line.startswith("运通"):
        log("INFO", "非运通线路不允许 4 位自编号，丢弃: {}".format(number), file_name, sp, tp)
        number = "unknown"
        flag = True
    # 验证模式：对比新旧结果，不一致时用新结果替换（unknown 保留旧值）
    if verify_mode:
        changed = False
        if line != "unknown" and line != old_line:
            log("INFO", "线路号变化: {} -> {}".format(old_line, line), file_name, sp, tp)
            changed = True
        elif line == "unknown":
            line = old_line
        if number != "unknown" and number != old_number:
            log("INFO", "自编号变化: {} -> {}".format(old_number, number), file_name, sp, tp)
            changed = True
        elif number == "unknown" and old_number:
            number = old_number
        if id_ != "unknown" and id_ != old_id:
            log("INFO", "车牌号变化: {} -> {}".format(old_id, id_), file_name, sp, tp)
            changed = True
        elif id_ == "unknown" and old_id:
            id_ = old_id
        if not changed:
            log("INFO", "验证通过，结果一致", file_name, sp, tp)
            return
        log("INFO", "验证未通过，使用新结果重命名", file_name, sp, tp)
        flag = False
        if number == "unknown" or id_ == "unknown":
            flag = True
    # 所有字段均为 unknown 时跳过重命名
    if line == "unknown" and number == "unknown" and id_ == "unknown":
        log("WARN", "所有字段均为 unknown，跳过重命名", file_name, sp, tp)
        return
    # 提取原始文件名（如果已重命名过，取最后一个 _ 后面的部分）
    original_name = file_name.split(".")[0]
    if re.match(r'^.+?路', file_name) and "_" in original_name:
        original_name = original_name.rsplit("_", 1)[-1]
    if re.match(non_bpt_line_regex, line) is not None:
        # 非公交集团线路用车牌号
        if flag is True:
            new_file_name = "{}路{}_{}.jpg".format(line, id_, original_name)
        else:
            new_file_name = "{}路{}.jpg".format(line, id_)
    else:
        if flag is True:
            new_file_name = "{}路{}_{}_{}.jpg".format(line, number, id_, original_name)
        else:
            new_file_name = "{}路{}_{}.jpg".format(line, number, id_)
    try:
        os.rename(os.path.join(file_path, file_name), os.path.join(file_path, new_file_name))
    except FileExistsError:
        new_file_name = "{}路{}_{}_{}.jpg".format(line, number, id_, original_name)
        os.rename(os.path.join(file_path, file_name), os.path.join(file_path, new_file_name))
    log("INFO", "{} -> {}".format(file_name, new_file_name), file_name, sp, tp)


def binary_image(image: Image, threshold: int = 128):
    """对图像进行二值化处理。"""
    table = [0 if i < threshold else 1 for i in range(256)]
    return image.point(table, "1")
