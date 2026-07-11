import datetime
import threading


class ProcessHandler:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self._lock = threading.Lock()

    def step_on(self):
        with self._lock:
            self.current_step += 1

    @property
    def process(self):
        with self._lock:
            return 100 * self.current_step / self.total_steps

    @property
    def steps(self):
        return self.total_steps

    @steps.setter
    def steps(self, value):
        self.total_steps = value


total_process_handler = ProcessHandler(1)

_log_lock = threading.Lock()


def log(level: str, message: str, file_name: str = "", single: float = -1, total: float = -1):
    """统一格式的日志输出。

    格式：[HH:MM:SS][LEVEL][单图进度%|总进度%] 文件名 - 消息
    level: INFO / WARN / ERROR
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with _log_lock:
        parts = ["[{}][{}]".format(timestamp, level)]
        if single >= 0:
            parts.append("[{:.2f}%|{:.2f}%]".format(single, total))
        if file_name:
            parts.append("{} -".format(file_name))
        parts.append(message)
        print(" ".join(parts))
