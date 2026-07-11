import re
import threading
import uuid
from threading import Thread

import numpy as np
from PIL import Image

from boat import Boat
from process import ProcessHandler, total_process_handler
from regex import bpt_line_regex, non_bpt_line_regex, number_regex, id_regex

print_lock = threading.Lock()
boat_lock = threading.Lock()


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
            with print_lock:
                print("[{} {:.2f}% {:.2f}%]OCR 失败：{}".format(self.file_path,
                                                                self.process_handler.process,
                                                                total_process_handler.process, e))
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
    # 生成唯一标识用于日志追踪
    log_id = uuid.uuid4().hex[:8]
    with print_lock:
        print("[{} {:.2f}% {:.2f}%]ocr variant: {}".format(file_path,
                                                            process_handler.process,
                                                            total_process_handler.process,
                                                            log_id))
    number_list = []
    line_list = []
    id_list = []
    # 直接将 PIL 图像转为 numpy 数组传给 PaddleOCR，省去磁盘 IO
    if image.mode == "1":
        image = image.convert("L")
    img_array = np.array(image)
    with print_lock:
        print("[{} {:.2f}% {:.2f}%]Begin OCR: {}".format(file_path,
                                                         process_handler.process,
                                                         total_process_handler.process, log_id))
    with boat_lock:
        boat = Boat()
        result = boat.paddle.ocr(img_array)
    with print_lock:
        print("[{} {:.2f}% {:.2f}%]Finish OCR: {}".format(file_path,
                                                          process_handler.process,
                                                          total_process_handler.process, log_id))
    result = result[0]
    for r in result:
        box = r[0]
        score = r[1][1]
        text = r[1][0]
        box_center = [int(sum([b[0] for b in box]) / 4 / image.width * 100),
                      int(sum([b[1] for b in box]) / 4 / image.height * 100)]
        with print_lock:
            print("[{} {:.2f}% {:.2f}%]{} text: {}, score: {:.4f}, box: {}".format(file_path,
                                                                                   process_handler.process,
                                                                                   total_process_handler.process,
                                                                                   log_id,
                                                                                   text,
                                                                                   score, box_center))
        if score < 0.75:
            continue
        if box_center[0] < 15 or box_center[0] > 85 or box_center[1] < 15 or box_center[1] > 85:
            continue

        id_temp = re.findall(id_regex, text)
        if len(id_temp) > 0:
            with print_lock:
                print("[{} {:.2f}% {:.2f}%]疑似车牌号: {}".format(
                    file_path,
                    process_handler.process,
                    total_process_handler.process,
                    id_temp))
            id_list.append((id_temp[0], score))
        else:
            number_temp = re.findall(number_regex, text)
            if len(number_temp) == 1 and len(number_temp[0]) > 0:
                with print_lock:
                    print("[{} {:.2f}% {:.2f}%]疑似自编号： {}".format(
                        file_path,
                        process_handler.process,
                        total_process_handler.process, number_temp))
                number_list.append((number_temp[0], score))
            else:
                bpt_line_temp = re.findall(bpt_line_regex, text)
                non_bpt_line_temp = re.findall(non_bpt_line_regex, text)
                if len(non_bpt_line_temp) == 1 and non_bpt_line_temp[0] != "0":
                    with print_lock:
                        print("[{} {:.2f}% {:.2f}%]疑似非公交集团线路号：{}".format(
                            file_path,
                            process_handler.process,
                            total_process_handler.process,
                            non_bpt_line_temp))
                    line_list.append((non_bpt_line_temp[0], score))
                if len(bpt_line_temp) == 1 and bpt_line_temp[0] != "0":
                    with print_lock:
                        print("[{} {:.2f}% {:.2f}%]疑似线路号：{}".format(
                            file_path,
                            process_handler.process,
                            total_process_handler.process,
                            bpt_line_temp))
                    line_list.append((bpt_line_temp[0], score))

    return line_list, number_list, id_list
