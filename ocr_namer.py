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
# 同位置疑似漏字读法的惩罚系数（见 penalize_missing_char_line）
MISSING_CHAR_PENALTY = 0.5
# 线路号提前退出所需的领先倍数：最高票不足次高票的这个倍数时视为分歧过大，继续取证
LINE_LEAD_RATIO = 1.5


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
    """检查是否已有足够的检测结果用于置信度投票（每个字段至少 2 个）。

    数量够了还要求线路号最高票明显领先：候选数量凑够 2 个但取值互相矛盾时，说明证据
    不足而不是充分，此时提前退出容易让微弱领先的错误读法定案。
    口径与最终投票保持一致：先按位置聚合、施加漏字与单位数惩罚，再比较前两名。
    """
    if not (len(line_list) >= 2 and len(number_list) >= 2 and len(id_list) >= 2):
        return False
    aggregated, _ = penalize_missing_char_line(aggregate_line_by_position(line_list))
    weighted = defaultdict(float)
    for text, score, _ in aggregated:
        weighted[text] += line_weight(text, score)
    if len(weighted) < 2:
        return True
    top_two = sorted(weighted.values(), reverse=True)[:2]
    return top_two[0] >= top_two[1] * LINE_LEAD_RATIO


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
    """把加权投票结果格式化为"值=权重(次数)"的降序字符串，便于排查误判来源。

    保留 4 位小数：这类误判常是零点几个百分点定胜负，2 位小数会把两个候选显示成同一
    个值，看不出真实差距。
    """
    items = sorted(weighted.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return " ".join("{}={:.4f}({}次)".format(text, weight, counts[text]) for text, weight in items)


def pick_winner(weighted: dict) -> str:
    """从加权投票结果中选出胜者，权重相同时优先更长的取值。

    不能直接用 max()：平票时它返回先插入的键，而插入顺序取决于哪个 OCR 线程先返回，
    同一张图重跑可能得到不同结果。平票时偏向更长取值，是因为 OCR 漏字比凭空多字常见。
    """
    return max(weighted, key=lambda text: (weighted[text], len(text), text))


def aggregate_line_by_position(line_list: list, tolerance: float = LINE_POSITION_TOLERANCE) -> list:
    """把同一线路号在同一位置的多次识别聚合成一条候选，重复次数按对数加成。

    21 个图像变体会把同一处文本反复识别，直接累加置信度等于把同一份证据数了 N 遍：
    车身碎片数字只要在多个通道图里稳定出现，就能压过只在少数变体里读出的真实
    LED 线路号。
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


def _is_subsequence(short: str, long: str) -> bool:
    """判断 short 是否为 long 的子序列（按顺序逐字符匹配，允许跳过）。

    用于识别 OCR 漏字：子串判断抓不到"中间丢字"这种最常见的漏读形态，只有子序列
    才能把丢掉中间字符的读法与完整读法关联起来。
    """
    it = iter(long)
    return all(any(ch == long_ch for long_ch in it) for ch in short)


def penalize_missing_char_line(line_list: list, tolerance: float = LINE_POSITION_TOLERANCE) -> list:
    """同一位置出现互斥读法时，对疑似漏字的较短候选降权。

    同一个位置只可能有一个真值，而 OCR 漏字远比凭空多字常见。因此同位置候选中，
    若短的是长的子序列，判定为漏字读法并降权。
    只降权不删除：万一短读法才对（如屏幕光斑被读成多余数字），它在多个变体一致出现
    时仍可凭重复加成翻盘。
    返回降权后的候选列表，并附带被降权的取值列表用于日志。
    """
    penalized = []
    demoted = []
    for index, (text, score, box) in enumerate(line_list):
        factor = 1.0
        if box is not None:
            cx, cy = _box_center(box)
            for other_index, (other_text, _, other_box) in enumerate(line_list):
                if other_index == index or other_box is None:
                    continue
                if len(other_text) <= len(text) or not _is_subsequence(text, other_text):
                    continue
                other_cx, other_cy = _box_center(other_box)
                if abs(cx - other_cx) <= tolerance and abs(cy - other_cy) <= tolerance:
                    factor = MISSING_CHAR_PENALTY
                    demoted.append("{}<-{}".format(text, other_text))
                    break
        penalized.append((text, score * factor, box))
    return penalized, demoted


def line_weight(text: str, score: float) -> float:
    """计算线路号候选的投票权重，对 1 位纯数字候选降权。"""
    if len(text) == 1 and text.isdigit():
        return score * SINGLE_DIGIT_LINE_PENALTY
    return score


def _determine_verify_mode(file_name: str, skip_named: bool):
    """判断文件是否进入验证模式。

    返回 (verify_mode, skip)：
    - verify_mode：文件名已是合法命名（无 unknown、线路号 ≥2 位），需重新 OCR 核对
    - skip：文件已命名且 skip_named=True，调用方应跳过该文件
    """
    line_prefix_match = re.match(r'^(.+?)路', file_name)
    if line_prefix_match:
        prefix = line_prefix_match.group(1)
        if re.fullmatch(bpt_line_regex, prefix) or re.fullmatch(non_bpt_line_regex, prefix):
            if "unknown" not in file_name and len(prefix) > 1:
                if skip_named:
                    return False, True
                return True, False
            log("INFO", "重新识别（unknown 或线路号为 1 位数）", file_name)
    return False, False


def _load_image(file_path: str, file_name: str, sp: float, tp: float):
    """加载图片并读取 EXIF 拍摄时间。

    返回 (image, exif_date)，加载失败时返回 (None, None) 并记录日志。
    """
    try:
        img = Image.open(os.path.join(file_path, file_name))
        # 必须在 convert("RGB") 之前读取 EXIF，convert 会丢弃元数据
        exif_date = _get_exif_datetime(img)
        image = img.convert("RGB")
    except Exception as e:
        log("ERROR", "图片加载失败，跳过: {}".format(e), file_name, sp, tp)
        return None, None
    if exif_date is not None:
        log("INFO", "拍摄时间: {}".format(exif_date), file_name, sp, tp)
    return image, exif_date


def _run_phases(file_name, base_images, verify_mode, single_process_handler):
    """执行三阶段递进 OCR，返回合并后的三类候选列表。

    每阶段结果不足时才进入下一阶段；验证模式强制跑完全部阶段以尽量多取证。
    """
    line_list, number_list, id_list = [], [], []

    def _sp():
        return single_process_handler.process
    def _tp():
        return total_process_handler.process

    # Phase 1：基础图像（多线程并行）
    l, n, i = run_ocr_batch(file_name, base_images, single_process_handler)
    line_list += l; number_list += n; id_list += i
    log("INFO", "Phase 1 完成 | 线路号 {} 个, 自编号 {} 个, 车牌号 {} 个".format(
        len(line_list), len(number_list), len(id_list)), file_name, _sp(), _tp())

    # Phase 2：RGB 通道拆分（结果不足时执行；验证模式强制执行以尽量多取证）
    # 验证是在核对已命名文件，需比新建更谨慎，故不走提前退出，
    # 多跑几种预处理（如红色通道能让红色 LED 点阵的细笔画更突出）。
    if verify_mode or not has_enough_results(line_list, number_list, id_list):
        channel_images = []
        for im in base_images:
            r, g, b = im.split()
            channel_images.extend([r, g, b])
        l, n, i = run_ocr_batch(file_name, channel_images, single_process_handler)
        line_list += l; number_list += n; id_list += i
        log("INFO", "Phase 2 完成 | 线路号 {} 个, 自编号 {} 个, 车牌号 {} 个".format(
            len(line_list), len(number_list), len(id_list)), file_name, _sp(), _tp())

    # Phase 3：Otsu 自适应二值化（结果不足时执行；验证模式强制执行以尽量多取证）
    if verify_mode or not has_enough_results(line_list, number_list, id_list):
        binary_images = []
        for im in base_images:
            r, g, b = im.split()
            for image_mono in [r, g, b]:
                threshold = otsu_threshold(np.array(image_mono))
                binary_images.append(binary_image(image_mono, threshold))
        l, n, i = run_ocr_batch(file_name, binary_images, single_process_handler)
        line_list += l; number_list += n; id_list += i
        log("INFO", "Phase 3 完成 | 线路号 {} 个, 自编号 {} 个, 车牌号 {} 个".format(
            len(line_list), len(number_list), len(id_list)), file_name, _sp(), _tp())

    return line_list, number_list, id_list


def _filter_number_by_date(number_list, exif_date, file_name, sp, tp):
    """按拍摄时间过滤自编号位数：2018 年后需 7 位，2026 年后仅允许 7 位纯数字。"""
    if exif_date is None:
        return number_list
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
    return new_number_list


def _find_overlap_container(text, box, line_counts, candidate_boxes):
    """在候选值容器（自编号/车牌号/其他线路号）中查找 text 的包含者。

    当 text 是某个更长候选的子串、出现次数不超过该候选、且 box 位置重叠时，判定为
    内嵌片段，返回 (container_text, container_count)；否则返回 None。
    box 为 None 时无法判断重叠，保守保留（返回 None）。
    """
    if box is None:
        return None
    for c_text, c_boxes in candidate_boxes.items():
        if len(text) < len(c_text) and text in c_text \
                and line_counts[text] <= len(c_boxes):
            if any(_box_overlap(box, cb) for cb in c_boxes if cb is not None):
                return c_text, len(c_boxes)
    return None


def _dedup_contained_lines(line_list, number_list, id_list, file_name, sp, tp):
    """确定性去重：线路号是其他更长候选值的子串、出现次数不多、且区域重叠时删除。

    box 不重叠说明是图中不同位置的文本，值包含只是数字巧合，不应删除。
    """
    line_counts = Counter(l[0] for l in line_list)
    # 容器候选（自编号、车牌号及更长线路号）：text -> [box1, box2, ...]
    containers = {}
    for text, _, box in number_list:
        containers.setdefault(text, []).append(box)
    for text, _, box in id_list:
        containers.setdefault(text, []).append(box)
    longer_lines = {}
    for text, _, box in line_list:
        longer_lines.setdefault(text, []).append(box)

    new_line_list = []
    for line_text, line_score, line_box in line_list:
        container = _find_overlap_container(
            line_text, line_box, line_counts, containers)
        if container is None:
            container = _find_overlap_container(
                line_text, line_box, line_counts, longer_lines)
        if container is not None:
            c_text, c_count = container
            log("INFO", "清理: 线路号 {} 包含在 {} 中且区域重叠 ({} 次 ≤ {} 次)".format(
                line_text, c_text, line_counts[line_text], c_count),
                file_name, sp, tp)
        else:
            new_line_list.append((line_text, line_score, line_box))
    return new_line_list


def _vote_line(line_list, file_name, sp, tp):
    """线路号置信度加权投票，返回 (胜者, 是否缺失)。空则胜者为 'unknown'。"""
    if not line_list:
        return "unknown", True
    weighted, counts = defaultdict(float), Counter()
    for text, score, _ in line_list:
        weighted[text] += line_weight(text, score)
        counts[text] += 1
    line = pick_winner(weighted)
    log("INFO", "线路号投票: {}".format(format_vote(weighted, counts)), file_name, sp, tp)
    return line, False


def _vote_field(candidates, label, file_name, sp, tp):
    """自编号/车牌号的置信度加权投票，返回 (胜者, 是否缺失)。空则胜者为 'unknown'。"""
    if not candidates:
        return "unknown", True
    weighted, counts = defaultdict(float), Counter()
    for text, score, _ in candidates:
        weighted[text] += score
        counts[text] += 1
    winner = pick_winner(weighted)
    log("INFO", "{}投票: {}".format(label, format_vote(weighted, counts)), file_name, sp, tp)
    return winner, False


def _vote_plate(id_list, number, file_name, sp, tp):
    """车牌号投票，返回 (胜者, 是否缺失)。

    车牌后缀先验：自编号第二位为 6 时大客车新能源牌多为「京A·5位数字D」，为 8 时
    多为「京A·5位数字F」。匹配到多个车牌时，对相应后缀车牌软性提权（仅提升权重，
    不保证胜出——存在多车场景，也有大客车挂黄色非新能源牌）。
    另优先选择京A开头的车牌号（北京公交集团车牌）。
    """
    if not id_list:
        return "unknown", True

    boost_suffix = None
    distinct_plates = set(item[0] for item in id_list)
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
    pool = jing_a_list if jing_a_list else id_list
    weighted, counts = defaultdict(float), Counter()
    for text, score, _ in pool:
        weighted[text] += plate_weight(text, score)
        counts[text] += 1
    id_ = pick_winner(weighted).replace("皖", "京")
    if jing_a_list:
        log("INFO", "优先选择京A车牌: {} | 投票: {}".format(
            id_, format_vote(weighted, counts)), file_name, sp, tp)
    else:
        log("INFO", "车牌号投票: {}".format(format_vote(weighted, counts)), file_name, sp, tp)
    return id_, False


def _resolve_verify_field(new_value, old_value, label, file_name, sp, tp):
    """验证模式下比对单个字段的新旧值。

    - 新值为 unknown：沿用旧值，不算变化
    - 新值是旧值的子串：视为 OCR 截断误读，保留旧值，不算变化
    - 其余不一致：记为变化
    返回 (采纳值, changed)。
    """
    if new_value == "unknown":
        return old_value, False
    if new_value != old_value:
        if old_value and new_value in old_value:
            log("INFO", "{} {} 是旧值 {} 的截断，保留旧值".format(
                label, new_value, old_value), file_name, sp, tp)
            return old_value, False
        log("INFO", "{}变化: {} -> {}".format(label, old_value, new_value), file_name, sp, tp)
        return new_value, True
    return new_value, False


def _resolve_verify(line, number, id_, old_line, old_number, old_id, file_name, sp, tp):
    """验证模式：对比新旧结果，不一致时用新结果替换。

    返回 (line, number, id_, changed, incomplete)。
    - changed=False 表示新旧一致，调用方应跳过重命名
    - incomplete=True 表示有字段缺失（unknown），需在文件名后追加原名
    """
    line, line_changed = _resolve_verify_field(line, old_line, "线路号", file_name, sp, tp)
    number, number_changed = _resolve_verify_field(number, old_number, "自编号", file_name, sp, tp)
    id_, id_changed = _resolve_verify_field(id_, old_id, "车牌号", file_name, sp, tp)

    changed = line_changed or number_changed or id_changed
    if not changed:
        log("INFO", "验证通过，结果一致", file_name, sp, tp)
        return line, number, id_, False, False
    log("INFO", "验证未通过，使用新结果重命名", file_name, sp, tp)
    incomplete = (number == "unknown" or id_ == "unknown")
    return line, number, id_, True, incomplete


def _build_new_name(line, number, id_, flag, original_name, file_name):
    """根据识别结果构造新文件名。"""
    if re.match(non_bpt_line_regex, line) is not None:
        # 非公交集团线路用车牌号
        if flag:
            return "{}路{}_{}.jpg".format(line, id_, original_name)
        return "{}路{}.jpg".format(line, id_)
    if flag:
        return "{}路{}_{}_{}.jpg".format(line, number, id_, original_name)
    return "{}路{}_{}.jpg".format(line, number, id_)


def _rename(file_path, file_name, new_file_name, line, number, id_, original_name, sp, tp):
    """重命名文件，目标名冲突时在末尾追加原名后重试。"""
    try:
        os.rename(os.path.join(file_path, file_name), os.path.join(file_path, new_file_name))
    except FileExistsError:
        # 冲突时附加原始文件名再试一次（与非公交集团线路分支格式一致）
        new_file_name = "{}路{}_{}_{}.jpg".format(line, number, id_, original_name)
        os.rename(os.path.join(file_path, file_name), os.path.join(file_path, new_file_name))
    log("INFO", "{} -> {}".format(file_name, new_file_name), file_name, sp, tp)


def ocr_namer(file_path: str, file_name: str, skip_named: bool = False):
    """对单张公交车照片执行 OCR 识别并重命名。

    采用三阶段递进策略，在检测结果充足时提前退出，避免无谓的 OCR 调用：
    Phase 1: 3 张基础图像（原图 / 模糊锐化 / 缩放锐化）
    Phase 2: 9 张 RGB 通道拆分（仅在 Phase 1 不足时）
    Phase 3: 9 张 Otsu 自适应二值化（仅在 Phase 2 不足时）
    每阶段内的图像变体通过多线程并行处理。

    对已命名文件：重新 OCR 验证，如果结果与文件名不一致则用新结果替换。
    """
    single_process_handler = ProcessHandler(MAX_STEPS)
    sp = single_process_handler.process
    tp = total_process_handler.process

    verify_mode, skip = _determine_verify_mode(file_name, skip_named)
    if skip:
        log("INFO", "跳过已命名文件", file_name)
        return

    if verify_mode:
        old_line, old_number, old_id = parse_filename(file_name)
        log("INFO", "验证模式 | 旧: 线路={}, 自编={}, 车牌={}".format(
            old_line, old_number, old_id), file_name, sp, tp)
    else:
        log("INFO", "开始处理 | 目录: {}".format(file_path), file_name, sp, tp)

    image, exif_date = _load_image(file_path, file_name, sp, tp)
    if image is None:
        return
    base_images = [
        image,
        image.filter(ImageFilter.GaussianBlur(radius=2)).filter(ImageFilter.EDGE_ENHANCE),
        image.resize((1280, 720)).filter(ImageFilter.GaussianBlur(radius=2)).filter(ImageFilter.EDGE_ENHANCE),
    ]

    line_list, number_list, id_list = _run_phases(
        file_name, base_images, verify_mode, single_process_handler)
    sp = single_process_handler.process
    tp = total_process_handler.process
    log("INFO", "疑似: 线路号={}, 自编号={}, 车牌号={}".format(
        [x[0] for x in line_list], [x[0] for x in number_list], [x[0] for x in id_list]),
        file_name, sp, tp)

    number_list = _filter_number_by_date(number_list, exif_date, file_name, sp, tp)

    line_list = _dedup_contained_lines(line_list, number_list, id_list, file_name, sp, tp)
    log("INFO", "清理后: 线路号={}, 自编号={}, 车牌号={}".format(
        [x[0] for x in line_list], [x[0] for x in number_list], [x[0] for x in id_list]),
        file_name, sp, tp)

    # 按位置聚合：同一位置被多个图像变体重复识别的线路号合并为一票（重复次数对数加成）
    before_aggregate = len(line_list)
    line_list = aggregate_line_by_position(line_list)
    if len(line_list) < before_aggregate:
        log("INFO", "线路号按位置聚合: {} -> {} 个，剩余={}".format(
            before_aggregate, len(line_list), [x[0] for x in line_list]), file_name, sp, tp)
    # 同位置互斥读法仲裁：疑似漏字的较短候选降权
    line_list, demoted_lines = penalize_missing_char_line(line_list)
    if demoted_lines:
        log("INFO", "疑似漏字读法降权: {}".format(" ".join(demoted_lines)), file_name, sp, tp)

    # 置信度加权投票：累计每个候选值的置信度，取最高者
    flag = False
    line, missing = _vote_line(line_list, file_name, sp, tp)
    flag = flag or missing
    number, missing = _vote_field(number_list, "自编号", file_name, sp, tp)
    flag = flag or missing
    id_, missing = _vote_plate(id_list, number, file_name, sp, tp)
    flag = flag or missing

    # 运通线路才允许 4 位自编号
    if number != "unknown" and len(number) == 4 and not line.startswith("运通"):
        log("INFO", "非运通线路不允许 4 位自编号，丢弃: {}".format(number), file_name, sp, tp)
        number = "unknown"
        flag = True

    # 验证模式：对比新旧结果，不一致时用新结果替换（unknown 保留旧值）
    # 截断保护：新识别值是旧值的子串时，视为 OCR 截断误读，保留已命名的旧值，不降级。
    if verify_mode:
        line, number, id_, changed, incomplete = _resolve_verify(
            line, number, id_, old_line, old_number, old_id, file_name, sp, tp)
        if not changed:
            return
        flag = incomplete

    # 所有字段均为 unknown 时跳过重命名
    if line == "unknown" and number == "unknown" and id_ == "unknown":
        log("WARN", "所有字段均为 unknown，跳过重命名", file_name, sp, tp)
        return

    # 提取原始文件名（如果已重命名过，取最后一个 _ 后面的部分）
    original_name = file_name.split(".")[0]
    if re.match(r'^.+?路', file_name) and "_" in original_name:
        original_name = original_name.rsplit("_", 1)[-1]

    new_file_name = _build_new_name(line, number, id_, flag, original_name, file_name)
    _rename(file_path, file_name, new_file_name, line, number, id_, original_name, sp, tp)


def binary_image(image: Image, threshold: int = 128):
    """对图像进行二值化处理。"""
    table = [0 if i < threshold else 1 for i in range(256)]
    return image.point(table, "1")
