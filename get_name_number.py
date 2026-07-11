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

# 运通线路号完整模式（用于跨文本区域合并）
yuntong_regex = re.compile(r'运通1[012][0-9]|运通20[0-9]')


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
    """
    log_id = uuid.uuid4().hex[:8]
    sp = process_handler.process
    tp = total_process_handler.process
    log("INFO", "OCR 变体 {}".format(log_id), file_path, sp, tp)
    number_list = []
    line_list = []
    id_list = []
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
            # 计算文本框中心位置（百分比）
            num_points = len(polys)
            box_center = [int(sum(p[0] for p in polys) / num_points / image.width * 100),
                          int(sum(p[1] for p in polys) / num_points / image.height * 100)]
            log("INFO", "text={}, score={:.4f}, box={}".format(text, score, box_center),
                file_path, sp, tp)
            # PaddleOCR 3.x 已内置 text_rec_score_thresh=0.75 过滤，此处仅做位置过滤
            if box_center[0] < 15 or box_center[0] > 85 or box_center[1] < 15 or box_center[1] > 85:
                continue

            # 收集所有通过过滤的文本片段（用于后续运通合并）
            all_texts.append(text)
            all_scores.append(score)

            id_temp = re.findall(id_regex, text)
            if len(id_temp) > 0:
                log("INFO", "疑似车牌号: {}".format(id_temp), file_path, sp, tp)
                id_list.append((id_temp[0], score))
            else:
                number_temp = re.findall(number_regex, text)
                if len(number_temp) == 1 and len(number_temp[0]) > 0:
                    log("INFO", "疑似自编号: {}".format(number_temp), file_path, sp, tp)
                    number_list.append((number_temp[0], score))
                else:
                    bpt_line_temp = re.findall(bpt_line_regex, text)
                    non_bpt_line_temp = re.findall(non_bpt_line_regex, text)
                    if len(non_bpt_line_temp) == 1 and non_bpt_line_temp[0] != "0":
                        log("INFO", "疑似非公交集团线路号: {}".format(non_bpt_line_temp), file_path, sp, tp)
                        line_list.append((non_bpt_line_temp[0], score))
                    if len(bpt_line_temp) == 1 and bpt_line_temp[0] != "0":
                        log("INFO", "疑似线路号: {}".format(bpt_line_temp), file_path, sp, tp)
                        line_list.append((bpt_line_temp[0], score))

    # 运通线路号合并：OCR 可能将"运通"和数字拆分为不同文本区域
    # 甚至"运"和"通"也分开，拼接所有文本后重新搜索
    if all_texts:
        joined_text = ''.join(all_texts)
        existing_lines = [l[0] for l in line_list]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.9
        yuntong_matches = re.findall(yuntong_regex, joined_text)
        for match in yuntong_matches:
            if match not in existing_lines:
                log("INFO", "合并运通线路号: {}".format(match), file_path, sp, tp)
                line_list.append((match, avg_score))

    return line_list, number_list, id_list
