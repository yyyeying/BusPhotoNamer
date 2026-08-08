import datetime
import math
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

# 车牌后缀先验加权系数：自编号第二位与新能源牌尾字母相关时，
# 对相应后缀车牌软性提权（仅提升权重，不保证胜出）
PLATE_SUFFIX_BOOST = 2.0

# 1 位纯数字线路号惩罚系数：北京确实有 5 路、7 路等单数字线路，但车身零碎数字、
# 车型标识被误读成 1 位数的概率远高于真实 LED 屏，故软降权而非直接丢弃。
SINGLE_DIGIT_LINE_PENALTY = 0.4
# 线路号按位置去重的容差（占图宽/高百分比）：中心点相距在此范围内视为同一处文本
LINE_POSITION_TOLERANCE = 3.0
# 同一位置被 N 个图像变体重复识别的加成系数：权重 = 最高分 × (1 + 系数 × ln(N))。
# 取对数而非线性累加：多变体一致仍算可靠性信号，但不能让同一份证据被数 N 遍。
REPEAT_BONUS = 0.5




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


def _get_exif_datetime(image: Image):
    """从图片 EXIF 中获取拍摄日期，失败或无 EXIF 时返回 None。

    优先读取 DateTimeOriginal（标签 36867），其次 DateTime（标签 306）。
    EXIF 时间格式通常为 'YYYY:MM:DD HH:MM:SS'。
    """
    try:
        exif = image.getexif()
        if not exif:
            return None
        dt_str = exif.get(36867) or exif.get(306)
        if not dt_str:
            return None
        return datetime.datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").date()
    except (KeyError, ValueError, AttributeError, TypeError):
        return None


def _box_overlap(box1: tuple, box2: tuple) -> bool:
    """判断两个边界框是否有交集。box = (x_min, y_min, x_max, y_max)"""
    return box1[0] < box2[2] and box2[0] < box1[2] and box1[1] < box2[3] and box2[1] < box1[3]


def _box_center(box: tuple) -> tuple:
    """返回边界框中心点 (cx, cy)。"""
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def format_vote(weighted: dict, counts: Counter) -> str:
    """把加权投票结果格式化为"值=权重(次数)"的降序字符串，便于排查误判来源。"""
    items = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)
    return " ".join("{}={:.2f}({}次)".format(text, weight, counts[text]) for text, weight in items)


def aggregate_line_by_position(line_list: list, tolerance: float = LINE_POSITION_TOLERANCE) -> list:
    """把同一线路号在同一位置的多次识别聚合成一条候选，重复次数按对数加成。

    21 个图像变体会把同一处文本反复识别，直接累加置信度等于把同一份证据数了 N 遍：
    车身碎片数字只要在多个通道图里稳定出现，就能压过只在少数变体里读出的真实 LED
    线路号（实测车头碎片"5"累计 3.60 分，压过只被读到 2 次的真实线路号）。
    但"多个变体读出同一结果"本身也是可靠性信号，所以不做简单去重，而是取该位置的
    最高分再按重复次数做次线性加成。不同位置的候选（如背景站牌）仍各自计票。
    仅对线路号生效：自编号/车牌的重复计票目前未观察到误判，保守不动。
    box 为 None 的汉字合并候选没有位置信息，原样保留。
    """
    # 每个分组为 [取值, 位置框, 该位置的所有置信度]
    groups = []
    for text, score, box in sorted(line_list, key=lambda item: item[1], reverse=True):
        if box is None:
            groups.append([text, box, [score]])
            continue
        cx, cy = _box_center(box)
        for group in groups:
            if group[0] != text or group[1] is None:
                continue
            group_cx, group_cy = _box_center(group[1])
            if abs(cx - group_cx) <= tolerance and abs(cy - group_cy) <= tolerance:
                group[2].append(score)
                break
        else:
            groups.append([text, box, [score]])
    return [(text, max(scores) * (1 + REPEAT_BONUS * math.log(len(scores))), box)
            for text, box, scores in groups]



def line_weight(text: str, score: float) -> float:
    """计算线路号候选的投票权重，对 1 位纯数字候选降权。"""
    if len(text) == 1 and text.isdigit():
        return score * SINGLE_DIGIT_LINE_PENALTY
    return score



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
    exif_date = None
    try:
        img = Image.open(os.path.join(file_path, file_name))
        # 必须在 convert("RGB") 之前读取 EXIF，convert 会丢弃元数据
        exif_date = _get_exif_datetime(img)
        image = img.convert("RGB")
    except Exception as e:
        log("ERROR", "图片加载失败，跳过: {}".format(e), file_name, sp, tp)
        return
    if exif_date is not None:
        log("INFO", "拍摄时间: {}".format(exif_date), file_name, sp, tp)
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

    # Phase 2：RGB 通道拆分（结果不足时执行；验证模式强制执行以尽量多取证）
    # 验证是在核对已命名文件，需比新建更谨慎，故不走提前退出，
    # 多跑几种预处理（如红色通道能让红色 LED 点阵的细笔画更突出）。
    if verify_mode or not has_enough_results(line_list, number_list, id_list):
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

    # Phase 3：Otsu 自适应二值化（结果不足时执行；验证模式强制执行以尽量多取证）
    if verify_mode or not has_enough_results(line_list, number_list, id_list):
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
    # 按拍摄时间过滤自编号位数：2018 年后需 7 位，2026 年后仅 7 位纯数字
    if exif_date is not None:
        new_number_list = []
        for num_text, num_score, num_box in number_list:
            if not num_text.isdigit():
                new_number_list.append((num_text, num_score, num_box))
                continue
            digit_count = len(num_text)
            if exif_date >= datetime.date(2026, 1, 1) and digit_count != 7:
                log("INFO", "拍摄时间 {} 起仅允许 7 位自编号，丢弃: {}".format(
                    exif_date, num_text), file_name, sp, tp)
            elif exif_date >= datetime.date(2018, 1, 1) and digit_count <= 6:
                log("INFO", "拍摄时间 {} 起不允许 6 位及以下自编号，丢弃: {}".format(
                    exif_date, num_text), file_name, sp, tp)
            else:
                new_number_list.append((num_text, num_score, num_box))
        number_list = new_number_list
    # 确定性去重：当线路号是其他更长候选值的子串、出现次数不超过容器、
    # 且 box 位置重叠时，删除该线路号候选。
    # box 不重叠说明是图中不同位置的文本，值包含只是数字巧合，不应删除。
    line_counts = Counter(l[0] for l in line_list)
    # 容器候选（自编号、车牌号）：text -> [box1, box2, ...]
    containers = {}
    for text, score, box in number_list:
        containers.setdefault(text, []).append(box)
    for text, score, box in id_list:
        containers.setdefault(text, []).append(box)
    # 线路号候选按 text 分组：text -> [box1, box2, ...]
    line_boxes_map = {}
    for text, score, box in line_list:
        line_boxes_map.setdefault(text, []).append(box)
    new_line_list = []
    for line_text, line_score, line_box in line_list:
        delete_flag = False
        container_text = ""
        container_count = 0
        # 检查是否是某个自编号/车牌号的子串
        for c_text, c_boxes in containers.items():
            if line_text in c_text and line_text != c_text:
                if line_counts[line_text] <= len(c_boxes):
                    # 需 box 重叠才删除（line_box 为 None 时无法判断，保守保留）
                    if line_box is not None and any(
                        _box_overlap(line_box, cb) for cb in c_boxes if cb is not None
                    ):
                        delete_flag = True
                        container_text = c_text
                        container_count = len(c_boxes)
                break
        # 检查是否是某个更长线路号的子串
        if delete_flag is False:
            for line2_text, line2_boxes in line_boxes_map.items():
                if len(line_text) < len(line2_text) and line_text in line2_text:
                    if line_counts[line_text] <= len(line2_boxes):
                        if line_box is not None and any(
                            _box_overlap(line_box, lb) for lb in line2_boxes if lb is not None
                        ):
                            delete_flag = True
                            container_text = line2_text
                            container_count = len(line2_boxes)
                    break
        if delete_flag is True:
            log("INFO", "清理: 线路号 {} 包含在 {} 中且区域重叠 ({} 次 ≤ {} 次)".format(
                line_text, container_text, line_counts[line_text], container_count),
                file_name, sp, tp)
        else:
            new_line_list.append((line_text, line_score, line_box))
    line_list = new_line_list
    log("INFO", "清理后: 线路号={}, 自编号={}, 车牌号={}".format(
        [x[0] for x in line_list], [x[0] for x in number_list], [x[0] for x in id_list]),
        file_name, sp, tp)
    # 按位置聚合：同一位置被多个图像变体重复识别的线路号合并为一票（重复次数对数加成）
    before_aggregate = len(line_list)
    line_list = aggregate_line_by_position(line_list)
    if len(line_list) < before_aggregate:
        log("INFO", "线路号按位置聚合: {} -> {} 个，剩余={}".format(
            before_aggregate, len(line_list), [x[0] for x in line_list]), file_name, sp, tp)


    # 置信度加权投票：累计每个候选值的置信度，取最高者
    flag = False
    if len(line_list) > 0:
        weighted = defaultdict(float)
        counts = Counter()
        for text, score, _ in line_list:
            weighted[text] += line_weight(text, score)
            counts[text] += 1
        line = max(weighted, key=weighted.get)
        log("INFO", "线路号投票: {}".format(format_vote(weighted, counts)), file_name, sp, tp)
    else:
        line = "unknown"
        flag = True
    if len(number_list) > 0:
        weighted = defaultdict(float)
        counts = Counter()
        for text, score, _ in number_list:
            weighted[text] += score
            counts[text] += 1
        number = max(weighted, key=weighted.get)
        log("INFO", "自编号投票: {}".format(format_vote(weighted, counts)), file_name, sp, tp)
    else:
        number = "unknown"
        flag = True

    if len(id_list) > 0:
        # 车牌后缀先验：自编号第二位为 6 时大客车新能源牌多为「京A·5位数字D」，
        # 为 8 时多为「京A·5位数字F」。匹配到多个车牌时，对相应后缀车牌软性提权。
        # 注意：仅提升权重不保证胜出——存在多车场景，也有大客车挂黄色非新能源牌。
        distinct_plates = set(item[0] for item in id_list)
        boost_suffix = None
        if (number != "unknown" and len(number) >= 2 and number[1].isdigit()
                and len(distinct_plates) >= 2):
            if number[1] == "6":
                boost_suffix = "D"
            elif number[1] == "8":
                boost_suffix = "F"

        def plate_weight(text, score):
            if boost_suffix and re.match(r'^京A·[0-9]{5}' + boost_suffix + r'$', text):
                return score * PLATE_SUFFIX_BOOST
            return score

        if boost_suffix:
            log("INFO", "自编号第二位={}，提升「京A·5位数字{}」车牌权重".format(
                number[1], boost_suffix), file_name, sp, tp)
        # 优先选择京A开头的车牌号（北京公交集团车牌）
        jing_a_list = [item for item in id_list if item[0].startswith("京A")]
        if jing_a_list:
            weighted = defaultdict(float)
            counts = Counter()
            for text, score, _ in jing_a_list:
                weighted[text] += plate_weight(text, score)
                counts[text] += 1
            id_ = max(weighted, key=weighted.get).replace("皖", "京")
            log("INFO", "优先选择京A车牌: {} | 投票: {}".format(
                id_, format_vote(weighted, counts)), file_name, sp, tp)
        else:
            weighted = defaultdict(float)
            counts = Counter()
            for text, score, _ in id_list:
                weighted[text] += plate_weight(text, score)
                counts[text] += 1
            id_ = max(weighted, key=weighted.get).replace("皖", "京")
            log("INFO", "车牌号投票: {}".format(format_vote(weighted, counts)), file_name, sp, tp)

    else:
        id_ = "unknown"
        flag = True
    # 运通线路才允许 4 位自编号
    if number != "unknown" and len(number) == 4 and not line.startswith("运通"):
        log("INFO", "非运通线路不允许 4 位自编号，丢弃: {}".format(number), file_name, sp, tp)
        number = "unknown"
        flag = True
    # 验证模式：对比新旧结果，不一致时用新结果替换（unknown 保留旧值）
    # 截断保护：新识别值是旧值的子串时，视为 OCR 截断误读（如"331"被读成"33"、
    # "4137289"被读成"413728"），保留已命名的旧值，不降级。
    if verify_mode:
        changed = False
        if line != "unknown" and line != old_line:
            if old_line and line in old_line:
                log("INFO", "线路号 {} 是旧值 {} 的截断，保留旧值".format(line, old_line), file_name, sp, tp)
                line = old_line
            else:
                log("INFO", "线路号变化: {} -> {}".format(old_line, line), file_name, sp, tp)
                changed = True
        elif line == "unknown":
            line = old_line
        if number != "unknown" and number != old_number:
            if old_number and number in old_number:
                log("INFO", "自编号 {} 是旧值 {} 的截断，保留旧值".format(number, old_number), file_name, sp, tp)
                number = old_number
            else:
                log("INFO", "自编号变化: {} -> {}".format(old_number, number), file_name, sp, tp)
                changed = True
        elif number == "unknown" and old_number:
            number = old_number
        if id_ != "unknown" and id_ != old_id:
            if old_id and id_ in old_id:
                log("INFO", "车牌号 {} 是旧值 {} 的截断，保留旧值".format(id_, old_id), file_name, sp, tp)
                id_ = old_id
            else:
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
