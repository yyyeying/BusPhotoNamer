import re
import threading
import uuid
from threading import Thread

import numpy as np
from PIL import Image

from boat import Boat
from process import ProcessHandler, log, total_process_handler
from regex import bpt_line_regex, non_bpt_line_regex, number_regex, id_regex

boat_lock = threading.Lock()


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

    每个列表的元素为 (文本, 置信度) 元组，置信度来自 OCR 识别分数。
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
    # 边界框信息：box=(x_min,y_min,x_max,y_max), area=面积
    line_boxes = []
    number_boxes = []
    id_boxes = []
    all_texts = []
    all_scores = []
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

            id_temp = re.findall(id_regex, text)
            if len(id_temp) > 0:
                log("INFO", "疑似车牌号: {}".format(id_temp), file_path, sp, tp)
                id_list.append((id_temp[0], score))
                id_ys.append(box_center[1])
                id_boxes.append((box, area))
            else:
                number_temp = re.findall(number_regex, text)
                if len(number_temp) == 1 and len(number_temp[0]) > 0:
                    log("INFO", "疑似自编号: {}".format(number_temp), file_path, sp, tp)
                    number_list.append((number_temp[0], score))
                    number_ys.append(box_center[1])
                    number_boxes.append((box, area))
                else:
                    bpt_line_temp = re.findall(bpt_line_regex, text)
                    non_bpt_line_temp = re.findall(non_bpt_line_regex, text)
                    if len(non_bpt_line_temp) == 1 and non_bpt_line_temp[0] != "0":
                        log("INFO", "疑似非公交集团线路号: {}".format(non_bpt_line_temp), file_path, sp, tp)
                        line_list.append((non_bpt_line_temp[0], score))
                        line_ys.append(box_center[1])
                        line_boxes.append((box, area))
                    if len(bpt_line_temp) == 1 and bpt_line_temp[0] != "0":
                        log("INFO", "疑似线路号: {}".format(bpt_line_temp), file_path, sp, tp)
                        line_list.append((bpt_line_temp[0], score))
                        line_ys.append(box_center[1])
                        line_boxes.append((box, area))

    # 位置置信度调整 1：线路号应在自编号上方，自编号应在车牌号上方
    # 违反位置关系时降低置信度（乘以 0.5）
    if line_ys and number_ys:
        avg_line_y = sum(line_ys) / len(line_ys)
        avg_number_y = sum(number_ys) / len(number_ys)
        if avg_line_y > avg_number_y:
            log("INFO", "位置异常: 线路号(y={:.0f})在自编号(y={:.0f})下方，降低线路号置信度".format(
                avg_line_y, avg_number_y), file_path, sp, tp)
            line_list = [(t, s * 0.5) for t, s in line_list]
    if number_ys and id_ys:
        avg_number_y = sum(number_ys) / len(number_ys)
        avg_id_y = sum(id_ys) / len(id_ys)
        if avg_number_y > avg_id_y:
            log("INFO", "位置异常: 自编号(y={:.0f})在车牌号(y={:.0f})下方，降低自编号置信度".format(
                avg_number_y, avg_id_y), file_path, sp, tp)
            number_list = [(t, s * 0.5) for t, s in number_list]

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
            line_list[i] = (line_list[i][0], line_list[i][1] * p)
    for i, p in enumerate(number_penalties):
        if p < 1.0:
            log("INFO", "区域重叠，降低自编号 '{}' 置信度".format(number_list[i][0]), file_path, sp, tp)
            number_list[i] = (number_list[i][0], number_list[i][1] * p)
    for i, p in enumerate(id_penalties):
        if p < 1.0:
            log("INFO", "区域重叠，降低车牌号 '{}' 置信度".format(id_list[i][0]), file_path, sp, tp)
            id_list[i] = (id_list[i][0], id_list[i][1] * p)

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
                    line_list.append((match, avg_score))
                    existing_lines.add(match)

    return line_list, number_list, id_list
