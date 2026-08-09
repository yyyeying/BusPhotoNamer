import re
import threading
import uuid
from collections import namedtuple
from threading import Thread

import numpy as np
from PIL import Image

from boat import Boat
from process import ProcessHandler, log, total_process_handler
from regex import bpt_line_regex, non_bpt_line_regex, number_regex, id_regex, plate_prefix_regex

boat_lock = threading.Lock()

# 线路号离群阈值：线路号框水平中心与主车锚点（车牌/自编号均值）的距离（占图宽百分比）
# 超过此值判定为背景/离群，软降权。取较大值以保守，只打击明显偏离画面边缘的候选。
# 不要再收紧：斜侧拍摄时车头 LED 与车牌的水平中心可差二十多个百分点，与背景路牌的
# 偏移量重叠，收紧会误伤真实线路号。背景干扰改由位置聚合+投票权重压制。
LINE_X_FAR_THRESHOLD = 30
# 离群线路号的惩罚系数：远离主车的候选（背景站牌/指路牌/其他车辆）乘以此系数。
# 取较小值以压制站牌等在多个变体里高频出现的背景线路号。
LINE_X_FAR_PENALTY = 0.3
# 线路号与自编号的最小垂直间距（占图高百分比）。线路号 LED 屏应明显高于车头自编号，
# 间距不足（含处于同一行）说明该数字是自编号旁的车身碎片，需降权。
MIN_LINE_NUMBER_GAP = 4
# 首位形近字母还原后，原截断候选的惩罚系数（见 _fix_digit_misread）
TRUNCATED_LINE_PENALTY = 0.6
# LED 点阵屏数字笔画残缺时被误读为形近字母的映射
DIGIT_MISREAD_MAP = {"I": "1", "L": "1", "T": "7", "Z": "2", "S": "5", "B": "8",
                     "G": "6", "A": "4", "J": "3", "O": "0", "D": "0", "Q": "0", "U": "0"}

# 单个 OCR 检测项：text=识别文本，score=置信度，box=归一化边界框，
# center=框中心 [cx, cy]（百分比），area=框面积。box 为 None 表示无位置信息（汉字合并候选）。
_Det = namedtuple('_Det', ['text', 'score', 'box', 'center', 'area'])


def _box_area(box: tuple) -> float:
    """计算边界框面积。box = (x_min, y_min, x_max, y_max)，单位为占图宽/高的百分比"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_overlap(box1: tuple, box2: tuple) -> float:
    """计算两个边界框的交集面积。"""
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    return (x_max - x_min) * (y_max - y_min)



def _normalize_plate(plate: str) -> str:
    """将车牌号归一化为 '省A·XXXXX' 格式，把分隔符统一为 ·。

    OCR 常把车牌圆点误读为 +、空格或直接省略，归一化后同一车牌的
    不同识别结果能合并计票。
    """
    m = re.match(r'([\u4e00-\u9fa5][A-Z])[·\-+ ]?([0-9A-Z]+)', plate)
    if m:
        return m.group(1) + "·" + m.group(2)
    return plate


def _fix_express_misread(text: str) -> str:
    """修正 LED 点阵屏"快"字被 OCR 误读为拉丁字母 k/K 的情况。

    仅在"数字+k+内/外"或"数字+内/外+k"的语境下替换，避免误伤其他 k。
    不替换的话，线路号正则会在字母 k 处截断，丢掉后面的"快"和方向字。
    """
    text = re.sub(r'(?<=\d)[kK](?=[内外])', '快', text)
    text = re.sub(r'(?<=\d[内外])[kK]', '快', text)
    return text


def _fix_digit_misread(text: str):
    """把"形近字母+2位数字"的线路号文本还原为 3 位线路号，无法还原时返回 None。

    LED 点阵屏笔画残缺时首位数字常被读成字母，正则会在字母处截断只得到后两位，
    丢掉一位。
    仅限还原后恰为 3 位的情况：数字部分已有 3 位的文本，其字母是屏边框/立柱噪声
    而不是数字，不能还原。
    """
    m = re.fullmatch(r'([A-Z])([0-9]{2})', text)
    if m is None:
        return None
    digit = DIGIT_MISREAD_MAP.get(m.group(1))
    if digit is None:
        return None
    fixed = digit + m.group(2)
    # 还原结果必须本身是合法线路号（首位为 0 的三位数不是）
    if re.fullmatch(bpt_line_regex, fixed) is None:
        return None
    return fixed



class GetNameNumber(Thread):
    def __init__(self, file_path: str, image: Image, process_handler: ProcessHandler):
        Thread.__init__(self)
        self.file_path = file_path
        self.image = image
        self.result = None
        self.process_handler = process_handler

    def run(self):
        try:
            self.result = self.get_name_number()
        except Exception as e:
            log("WARN", "OCR 失败: {}".format(e), self.file_path,
                self.process_handler.process, total_process_handler.process)
            self.result = None
        self.process_handler.step_on()
        total_process_handler.step_on()

    def get_result(self):
        return self.result

    def get_name_number(self):
        return get_name_number(self.file_path, self.image, self.process_handler)


def _make_det(text: str, score: float, polys, image: Image) -> _Det:
    """由 OCR 多边形构造归一化的检测记录（box/center/area 均为占图百分比）。

    box 用百分比而非像素：各图像变体尺寸不同（含 resize 到 1280x720 的变体），像素
    坐标跨变体无法比较，归一化后才能做跨变体的位置去重与重叠判断。
    """
    num_points = len(polys)
    x_coords = [p[0] for p in polys]
    y_coords = [p[1] for p in polys]
    center = [int(sum(x_coords) / num_points / image.width * 100),
              int(sum(y_coords) / num_points / image.height * 100)]
    box = (min(x_coords) / image.width * 100, min(y_coords) / image.height * 100,
           max(x_coords) / image.width * 100, max(y_coords) / image.height * 100)
    return _Det(text, score, box, center, _box_area(box))


def _classify_line_candidate(det: _Det, line_text: str, lines: list, file_path, sp, tp):
    """把一个非车牌/非自编号的文本片段判定为线路号候选并登记。

    处理"快"误读、公交/非公交集团正则匹配、数字前缀守卫、首位形近字母还原。
    可能登记 0、1 或 2 个线路号候选（还原时登记还原值 + 降权的原截断值）。
    """
    # 排除背景车辆车牌片段：以车牌前缀（省份+字母+数字，如"京A19"）开头的文本
    # 不是线路号，避免把未识别完整的车牌当成线路号
    if plate_prefix_regex.match(line_text):
        log("INFO", "疑似车牌前缀，不作为线路号: {}".format(line_text), file_path, sp, tp)
        return
    non_bpt_line_temp = re.findall(non_bpt_line_regex, line_text)
    bpt_matches = list(re.finditer(bpt_line_regex, line_text))
    if len(non_bpt_line_temp) == 1 and non_bpt_line_temp[0] != "0":
        log("INFO", "疑似非公交集团线路号: {}".format(non_bpt_line_temp), file_path, sp, tp)
        lines.append(det._replace(text=non_bpt_line_temp[0]))
    if len(bpt_matches) == 1 and bpt_matches[0].group() != "0":
        m = bpt_matches[0]
        # 数字前缀守卫：线路号数字前紧邻其他数字，说明它是更长数字（如车内编号）的一部分，拒绝
        if m.start() > 0 and line_text[m.start() - 1].isdigit():
            log("INFO", "疑似数字内嵌片段，不作为线路号: {}".format(line_text), file_path, sp, tp)
            return
        # 首位形近字母还原。原截断候选降权保留，万一字母确实是噪声（真实线路为 2 位），
        # 仍可靠其他变体读出的纯数字结果累计票数翻盘。
        fixed = _fix_digit_misread(line_text)
        if fixed is not None:
            log("INFO", "疑似首位误读: {} -> {}，原候选 {} 降权".format(
                line_text, fixed, m.group()), file_path, sp, tp)
            lines.append(det._replace(text=fixed))
            lines.append(det._replace(text=m.group(), score=det.score * TRUNCATED_LINE_PENALTY))
        else:
            log("INFO", "疑似线路号: {}".format([m.group()]), file_path, sp, tp)
            lines.append(det._replace(text=m.group()))


def _classify_detection(det: _Det, lines: list, numbers: list, ids: list, file_path, sp, tp):
    """把单个 OCR 文本片段分类为车牌号/自编号/线路号候选并登记。"""
    text = det.text
    id_temp = re.findall(id_regex, text)
    if len(id_temp) > 0:
        plate = _normalize_plate(id_temp[0])
        log("INFO", "疑似车牌号: {}".format(plate), file_path, sp, tp)
        ids.append(det._replace(text=plate))
        return
    number_temp = re.findall(number_regex, text)
    if len(number_temp) == 1 and len(number_temp[0]) > 0:
        log("INFO", "疑似自编号: {}".format(number_temp), file_path, sp, tp)
        numbers.append(det._replace(text=number_temp[0]))
        return
    # 修正 LED 屏"快"被误读为 k
    line_text = _fix_express_misread(text)
    _classify_line_candidate(det, line_text, lines, file_path, sp, tp)


def _merge_split_plates(raw_dets: list, ids: list, file_path, sp, tp):
    """拼接被 OCR 拆成前缀框 + 号码框的车牌，补入车牌候选。

    OCR 有时把车牌拆成省份+字母的前缀框和号码框两个（同一行、水平相邻）。单独的前缀
    太短、号码段会被当成自编号，导致主车真车牌丢失。此处把相邻的前缀+号码拼成完整
    车牌补入候选。
    """
    prefix_dets = [d for d in raw_dets if re.fullmatch(r'[\u4e00-\u9fa5][A-Z]', d.text)]
    tail_dets = [d for d in raw_dets
                 if re.fullmatch(r'[0-9A-Z]{5,6}[DF]?', d.text) and re.search(r'[0-9]', d.text)]
    for p in prefix_dets:
        for t in tail_dets:
            # 同一行、号码在前缀右侧且水平相邻
            if abs(p.center[1] - t.center[1]) <= 4 and -2 <= (t.center[0] - p.center[0]) <= 20:
                plate = _normalize_plate(p.text + t.text)
                merged_box = (min(p.box[0], t.box[0]), min(p.box[1], t.box[1]),
                              max(p.box[2], t.box[2]), max(p.box[3], t.box[3]))
                log("INFO", "拼接拆分车牌: {} + {} -> {}".format(p.text, t.text, plate),
                    file_path, sp, tp)
                ids.append(_Det(plate, min(p.score, t.score), merged_box, t.center,
                                _box_area(merged_box)))


def _check_vertical_position(lines: list, numbers: list, ids: list, file_path, sp, tp):
    """垂直位置校验：线路号应在自编号上方，自编号应在车牌号上方。

    逐候选判断（而非用整类平均值），避免底部离群项被顶部候选"平均"掩盖：
    车型型号里的数字位于车身底部，应单独降权，而顶部 LED 线路号不受影响。
    违反位置关系的候选置信度乘以 0.5。
    线路号还要求与自编号有 MIN_LINE_NUMBER_GAP 的垂直间距：只判"严格在下方"会漏掉
    与自编号同一行的车身碎片数字。
    """
    if lines and numbers:
        avg_number_y = sum(d.center[1] for d in numbers) / len(numbers)
        for idx, d in enumerate(lines):
            if d.center[1] > avg_number_y - MIN_LINE_NUMBER_GAP:
                log("INFO", "位置异常: 线路号 '{}'(y={:.0f})未明显高于自编号(y={:.0f})，降低置信度".format(
                    d.text, d.center[1], avg_number_y), file_path, sp, tp)
                lines[idx] = d._replace(score=d.score * 0.5)
    if numbers and ids:
        avg_id_y = sum(d.center[1] for d in ids) / len(ids)
        for idx, d in enumerate(numbers):
            if d.center[1] > avg_id_y:
                log("INFO", "位置异常: 自编号 '{}'(y={:.0f})在车牌号(y={:.0f})下方，降低置信度".format(
                    d.text, d.center[1], avg_id_y), file_path, sp, tp)
                numbers[idx] = d._replace(score=d.score * 0.5)


def _check_horizontal_position(lines: list, numbers: list, ids: list, file_path, sp, tp):
    """水平位置校验：线路号应靠近主车（车牌/自编号所在的水平区域）。

    背景指路牌/站牌/其他车辆上的数字常在画面边缘，其 x 水平中心明显偏离主车候选簇，
    对这类远离锚点的线路号软降权。锚点（车牌/自编号）缺失时不启用，保持保守。
    """
    anchor_xs = [d.center[0] for d in numbers] + [d.center[0] for d in ids]
    if not (anchor_xs and lines):
        return
    anchor_x = sum(anchor_xs) / len(anchor_xs)
    for idx, d in enumerate(lines):
        if abs(d.center[0] - anchor_x) > LINE_X_FAR_THRESHOLD:
            log("INFO", "位置异常: 线路号 '{}'(x={:.0f})远离主车(x={:.0f})，降低置信度".format(
                d.text, d.center[0], anchor_x), file_path, sp, tp)
            lines[idx] = d._replace(score=d.score * LINE_X_FAR_PENALTY)


def _check_region_overlap(lines: list, numbers: list, ids: list, file_path, sp, tp):
    """区域重叠校验：不同类型的检测区域不应重叠，重叠时面积较小的置信度乘以 0.5。

    同一候选与多个异类区域重叠时惩罚累乘。
    """
    line_pen = [1.0] * len(lines)
    number_pen = [1.0] * len(numbers)
    id_pen = [1.0] * len(ids)

    def _pair(a, a_pen, b, b_pen):
        for i, da in enumerate(a):
            for j, db in enumerate(b):
                if _box_overlap(da.box, db.box) > 0:
                    if da.area < db.area:
                        a_pen[i] *= 0.5
                    else:
                        b_pen[j] *= 0.5

    _pair(lines, line_pen, numbers, number_pen)
    _pair(lines, line_pen, ids, id_pen)
    _pair(numbers, number_pen, ids, id_pen)

    for i, p in enumerate(line_pen):
        if p < 1.0:
            log("INFO", "区域重叠，降低线路号 '{}' 置信度".format(lines[i].text), file_path, sp, tp)
            lines[i] = lines[i]._replace(score=lines[i].score * p)
    for i, p in enumerate(number_pen):
        if p < 1.0:
            log("INFO", "区域重叠，降低自编号 '{}' 置信度".format(numbers[i].text), file_path, sp, tp)
            numbers[i] = numbers[i]._replace(score=numbers[i].score * p)
    for i, p in enumerate(id_pen):
        if p < 1.0:
            log("INFO", "区域重叠，降低车牌号 '{}' 置信度".format(ids[i].text), file_path, sp, tp)
            ids[i] = ids[i]._replace(score=ids[i].score * p)


def _merge_chinese_char_lines(lines: list, all_texts: list, all_scores: list, file_path, sp, tp):
    """汉字前缀线路号合并：OCR 可能将汉字和数字拆分为不同文本区域。

    拼接所有文本后重新搜索，补充未被单区域匹配到的汉字线路号（box 为 None）。
    """
    if not all_texts:
        return
    joined_text = ''.join(all_texts)
    existing_lines = set(d.text for d in lines)
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.9
    for regex in [bpt_line_regex, non_bpt_line_regex]:
        for match in re.findall(regex, joined_text):
            # 仅添加含汉字且未被单区域匹配到的结果
            if match not in existing_lines and re.search(r'[\u4e00-\u9fa5]', match):
                log("INFO", "合并线路号: {}".format(match), file_path, sp, tp)
                lines.append(_Det(match, avg_score, None, None, 0.0))
                existing_lines.add(match)


def get_name_number(file_path: str, image: Image, process_handler: ProcessHandler):
    """对单个图像变体执行 OCR，返回疑似线路号、自编号、车牌号列表。

    每个列表的元素为 (文本, 置信度, box) 元组：
    - 文本：识别出的线路号/自编号/车牌号字符串
    - 置信度：OCR 识别分数，会根据空间位置关系调整
    - box：边界框 (x_min, y_min, x_max, y_max)，汉字合并候选为 None
    置信度会根据空间位置关系调整：
    1. 线路号应在自编号上方，自编号应在车牌号上方
    2. 不同类型的检测区域不应重叠，重叠时面积较小的降权
    """
    log_id = uuid.uuid4().hex[:8]
    sp = process_handler.process
    tp = total_process_handler.process
    log("INFO", "OCR 变体 {}".format(log_id), file_path, sp, tp)

    # PaddleOCR 3.x 要求 RGB 图像，灰度/二值图需转换
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.array(image)
    log("INFO", "Begin OCR {}".format(log_id), file_path, sp, tp)
    with boat_lock:
        boat = Boat()
        result = list(boat.paddle.predict(img_array))
    log("INFO", "Finish OCR {}".format(log_id), file_path, sp, tp)

    lines, numbers, ids = [], [], []
    all_texts, all_scores = [], []
    raw_dets = []
    for res in result:
        json_data = res.json['res']
        rec_texts = json_data["rec_texts"]
        rec_scores = json_data["rec_scores"]
        rec_polys = json_data["rec_polys"]
        for text, score, polys in zip(rec_texts, rec_scores, rec_polys):
            det = _make_det(text, score, polys, image)
            log("INFO", "text={}, score={:.4f}, box={}, area={:.2f}".format(
                text, score, det.center, det.area), file_path, sp, tp)
            # 收集所有文本片段（用于后续汉字线路号合并）
            all_texts.append(text)
            all_scores.append(score)
            raw_dets.append(det)
            _classify_detection(det, lines, numbers, ids, file_path, sp, tp)

    _merge_split_plates(raw_dets, ids, file_path, sp, tp)
    _check_vertical_position(lines, numbers, ids, file_path, sp, tp)
    _check_horizontal_position(lines, numbers, ids, file_path, sp, tp)
    _check_region_overlap(lines, numbers, ids, file_path, sp, tp)
    _merge_chinese_char_lines(lines, all_texts, all_scores, file_path, sp, tp)

    return ([(d.text, d.score, d.box) for d in lines],
            [(d.text, d.score, d.box) for d in numbers],
            [(d.text, d.score, d.box) for d in ids])
