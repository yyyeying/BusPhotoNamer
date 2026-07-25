import re
import threading
import uuid
from threading import Thread

import numpy as np
from PIL import Image

from boat import Boat
from process import ProcessHandler, log, total_process_handler
from regex import bpt_line_regex, non_bpt_line_regex, number_regex, id_regex, plate_prefix_regex

boat_lock = threading.Lock()

# 方案 J 阈值：线路号框水平中心与主车锚点（车牌/自编号均值）的距离（占图宽百分比）
# 超过此值判定为背景/离群，软降权。取较大值以保守，只打击明显偏离画面边缘的候选。
LINE_X_FAR_THRESHOLD = 30
# 方案 J 惩罚系数：远离主车的线路号候选（背景站牌/指路牌/其他车辆）乘以此系数。
# 取较小值以压制站牌等高频出现的背景线路号（如站牌"671"被读十余次）。
LINE_X_FAR_PENALTY = 0.3


def _box_area(box: tuple) -> int:
    """计算边界框面积。box = (x_min, y_min, x_max, y_max)"""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_overlap(box1: tuple, box2: tuple) -> int:
    """计算两个边界框的交集面积。"""
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    if x_max <= x_min or y_max <= y_min:
        return 0
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


def _fix_kuai_misread(text: str) -> str:
    """修正 LED 点阵屏"快"字被 OCR 误读为拉丁字母 k/K 的情况。

    仅在"数字+k+内/外"或"数字+内/外+k"的语境下替换，避免误伤其他 k。
    例如 "300k外" -> "300快外"，否则正则会在 k 处截断只得到 "300"。
    """
    text = re.sub(r'(?<=\d)[kK](?=[内外])', '快', text)
    text = re.sub(r'(?<=\d[内外])[kK]', '快', text)
    return text


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
    number_list = []
    line_list = []
    id_list = []
    line_ys = []
    number_ys = []
    id_ys = []
    # 各候选框的水平中心（百分比），用于方案 J 的水平聚类判断
    line_xs = []
    number_xs = []
    id_xs = []
    # 边界框信息：box=(x_min,y_min,x_max,y_max), area=面积
    line_boxes = []
    number_boxes = []
    id_boxes = []
    all_texts = []
    all_scores = []
    # 原始检测框（text, score, box, box_center），用于后续拆分车牌拼接
    raw_dets = []
    # PaddleOCR 3.x 要求 RGB 图像，灰度/二值图需转换
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.array(image)
    log("INFO", "Begin OCR {}".format(log_id), file_path, sp, tp)
    with boat_lock:
        boat = Boat()
        result = list(boat.paddle.predict(img_array))
    log("INFO", "Finish OCR {}".format(log_id), file_path, sp, tp)
    for res in result:
        json_data = res.json['res']
        rec_texts = json_data["rec_texts"]
        rec_scores = json_data["rec_scores"]
        rec_polys = json_data["rec_polys"]
        for text, score, polys in zip(rec_texts, rec_scores, rec_polys):
            # 计算文本框中心位置（百分比，Y 轴 0=顶部 100=底部）
            num_points = len(polys)
            x_coords = [p[0] for p in polys]
            y_coords = [p[1] for p in polys]
            box_center = [int(sum(x_coords) / num_points / image.width * 100),
                          int(sum(y_coords) / num_points / image.height * 100)]
            box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            area = _box_area(box)
            log("INFO", "text={}, score={:.4f}, box={}, area={}".format(text, score, box_center, area),
                file_path, sp, tp)

            # 收集所有文本片段（用于后续汉字线路号合并）
            all_texts.append(text)
            all_scores.append(score)
            raw_dets.append((text, score, box, box_center))

            id_temp = re.findall(id_regex, text)
            if len(id_temp) > 0:
                plate = _normalize_plate(id_temp[0])
                log("INFO", "疑似车牌号: {}".format(plate), file_path, sp, tp)
                id_list.append((plate, score, box))
                id_ys.append(box_center[1])
                id_xs.append(box_center[0])
                id_boxes.append((box, area))
            else:
                number_temp = re.findall(number_regex, text)
                if len(number_temp) == 1 and len(number_temp[0]) > 0:
                    log("INFO", "疑似自编号: {}".format(number_temp), file_path, sp, tp)
                    number_list.append((number_temp[0], score, box))
                    number_ys.append(box_center[1])
                    number_xs.append(box_center[0])
                    number_boxes.append((box, area))
                else:
                    # 排除背景车辆车牌片段：以车牌前缀（省份+字母+数字，如"京A19"）
                    # 开头的文本不是线路号，避免把未识别完整的车牌当成线路号
                    if plate_prefix_regex.match(text):
                        log("INFO", "疑似车牌前缀，不作为线路号: {}".format(text), file_path, sp, tp)
                    else:
                        # E：修正 LED 屏"快"被误读为 k（如"300k外"->"300快外"）
                        line_text = _fix_kuai_misread(text)
                        non_bpt_line_temp = re.findall(non_bpt_line_regex, line_text)
                        bpt_matches = list(re.finditer(bpt_line_regex, line_text))
                        if len(non_bpt_line_temp) == 1 and non_bpt_line_temp[0] != "0":
                            log("INFO", "疑似非公交集团线路号: {}".format(non_bpt_line_temp), file_path, sp, tp)
                            line_list.append((non_bpt_line_temp[0], score, box))
                            line_ys.append(box_center[1])
                            line_xs.append(box_center[0])
                            line_boxes.append((box, area))
                        if len(bpt_matches) == 1 and bpt_matches[0].group() != "0":
                            m = bpt_matches[0]
                            # F：数字前缀守卫——线路号数字前紧邻其他数字，
                            # 说明它是更长数字的一部分（如车内编号"0281"->"281"），拒绝
                            if m.start() > 0 and line_text[m.start() - 1].isdigit():
                                log("INFO", "疑似数字内嵌片段，不作为线路号: {}".format(line_text),
                                    file_path, sp, tp)
                            else:
                                log("INFO", "疑似线路号: {}".format([m.group()]), file_path, sp, tp)
                                line_list.append((m.group(), score, box))
                                line_ys.append(box_center[1])
                                line_xs.append(box_center[0])
                                line_boxes.append((box, area))

    # 拆分车牌拼接：OCR 有时把车牌拆成"京A"前缀框 + "48040F"号码框两个（同一行、
    # 水平相邻）。单独的前缀太短、号码段会被当成自编号，导致主车真车牌丢失。
    # 此处把相邻的前缀+号码拼成完整车牌补入候选。
    prefix_dets = [d for d in raw_dets if re.fullmatch(r'[\u4e00-\u9fa5][A-Z]', d[0])]
    tail_dets = [d for d in raw_dets
                 if re.fullmatch(r'[0-9A-Z]{5,6}[DF]?', d[0]) and re.search(r'[0-9]', d[0])]
    for p_text, p_score, p_box, p_center in prefix_dets:
        for t_text, t_score, t_box, t_center in tail_dets:
            # 同一行、号码在前缀右侧且水平相邻
            if abs(p_center[1] - t_center[1]) <= 4 and -2 <= (t_center[0] - p_center[0]) <= 20:
                plate = _normalize_plate(p_text + t_text)
                merged_box = (min(p_box[0], t_box[0]), min(p_box[1], t_box[1]),
                              max(p_box[2], t_box[2]), max(p_box[3], t_box[3]))
                log("INFO", "拼接拆分车牌: {} + {} -> {}".format(p_text, t_text, plate),
                    file_path, sp, tp)
                id_list.append((plate, min(p_score, t_score), merged_box))
                id_ys.append(t_center[1])
                id_xs.append(t_center[0])
                id_boxes.append((merged_box, _box_area(merged_box)))

    # 位置置信度调整 1：线路号应在自编号上方，自编号应在车牌号上方
    # 逐候选判断（而非用整类平均值），避免底部离群项被顶部候选"平均"掩盖：
    # 例如车型型号"C10E"里的"10"位于车身底部，应单独降权，
    # 而顶部 LED 线路号"606"不受影响。违反位置关系的候选置信度乘以 0.5。
    if line_ys and number_ys:
        avg_number_y = sum(number_ys) / len(number_ys)
        new_line_list = []
        for idx, (t, s, b) in enumerate(line_list):
            if line_ys[idx] > avg_number_y:
                log("INFO", "位置异常: 线路号 '{}'(y={:.0f})在自编号(y={:.0f})下方，降低置信度".format(
                    t, line_ys[idx], avg_number_y), file_path, sp, tp)
                new_line_list.append((t, s * 0.5, b))
            else:
                new_line_list.append((t, s, b))
        line_list = new_line_list
    if number_ys and id_ys:
        avg_id_y = sum(id_ys) / len(id_ys)
        new_number_list = []
        for idx, (t, s, b) in enumerate(number_list):
            if number_ys[idx] > avg_id_y:
                log("INFO", "位置异常: 自编号 '{}'(y={:.0f})在车牌号(y={:.0f})下方，降低置信度".format(
                    t, number_ys[idx], avg_id_y), file_path, sp, tp)
                new_number_list.append((t, s * 0.5, b))
            else:
                new_number_list.append((t, s, b))
        number_list = new_number_list

    # 位置置信度调整 J：线路号应靠近主车（车牌/自编号所在的水平区域）。
    # 背景指路牌/其他车辆的数字（如高速"41出口"牌、背景路牌"128"）常在画面边缘，
    # 其 x 水平中心明显偏离主车候选簇，对这类远离锚点的线路号软降权（乘以 0.5）。
    # 锚点（车牌/自编号）缺失时不启用，保持保守；仅按水平距离判断，不影响正常居中线路号。
    anchor_xs = number_xs + id_xs
    if anchor_xs and line_xs:
        anchor_x = sum(anchor_xs) / len(anchor_xs)
        new_line_list = []
        for idx, (t, s, b) in enumerate(line_list):
            if abs(line_xs[idx] - anchor_x) > LINE_X_FAR_THRESHOLD:
                log("INFO", "位置异常: 线路号 '{}'(x={:.0f})远离主车(x={:.0f})，降低置信度".format(
                    t, line_xs[idx], anchor_x), file_path, sp, tp)
                new_line_list.append((t, s * LINE_X_FAR_PENALTY, b))
            else:
                new_line_list.append((t, s, b))
        line_list = new_line_list

    # 位置置信度调整 2：不同类型的检测区域不应重叠
    # 重叠时面积较小的置信度乘以 0.5
    line_penalties = [1.0] * len(line_list)
    number_penalties = [1.0] * len(number_list)
    id_penalties = [1.0] * len(id_list)
    # 线路号 vs 自编号
    for i in range(len(line_boxes)):
        for j in range(len(number_boxes)):
            if _box_overlap(line_boxes[i][0], number_boxes[j][0]) > 0:
                if line_boxes[i][1] < number_boxes[j][1]:
                    line_penalties[i] *= 0.5
                else:
                    number_penalties[j] *= 0.5
    # 线路号 vs 车牌号
    for i in range(len(line_boxes)):
        for j in range(len(id_boxes)):
            if _box_overlap(line_boxes[i][0], id_boxes[j][0]) > 0:
                if line_boxes[i][1] < id_boxes[j][1]:
                    line_penalties[i] *= 0.5
                else:
                    id_penalties[j] *= 0.5
    # 自编号 vs 车牌号
    for i in range(len(number_boxes)):
        for j in range(len(id_boxes)):
            if _box_overlap(number_boxes[i][0], id_boxes[j][0]) > 0:
                if number_boxes[i][1] < id_boxes[j][1]:
                    number_penalties[i] *= 0.5
                else:
                    id_penalties[j] *= 0.5
    # 应用重叠惩罚
    for i, p in enumerate(line_penalties):
        if p < 1.0:
            log("INFO", "区域重叠，降低线路号 '{}' 置信度".format(line_list[i][0]), file_path, sp, tp)
            line_list[i] = (line_list[i][0], line_list[i][1] * p, line_list[i][2])
    for i, p in enumerate(number_penalties):
        if p < 1.0:
            log("INFO", "区域重叠，降低自编号 '{}' 置信度".format(number_list[i][0]), file_path, sp, tp)
            number_list[i] = (number_list[i][0], number_list[i][1] * p, number_list[i][2])
    for i, p in enumerate(id_penalties):
        if p < 1.0:
            log("INFO", "区域重叠，降低车牌号 '{}' 置信度".format(id_list[i][0]), file_path, sp, tp)
            id_list[i] = (id_list[i][0], id_list[i][1] * p, id_list[i][2])

    # 汉字前缀线路号合并：OCR 可能将汉字和数字拆分为不同文本区域
    # 拼接所有文本后重新搜索，补充未被单区域匹配到的汉字线路号
    if all_texts:
        joined_text = ''.join(all_texts)
        existing_lines = set(l[0] for l in line_list)
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.9
        for regex in [bpt_line_regex, non_bpt_line_regex]:
            for match in re.findall(regex, joined_text):
                # 仅添加含汉字且未被单区域匹配到的结果
                if match not in existing_lines and re.search(r'[\u4e00-\u9fa5]', match):
                    log("INFO", "合并线路号: {}".format(match), file_path, sp, tp)
                    line_list.append((match, avg_score, None))
                    existing_lines.add(match)

    return line_list, number_list, id_list
