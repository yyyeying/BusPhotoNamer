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
# 本次运行的日志文件句柄，由 setup_log_file() 设置、close_log_file() 关闭
_log_file = None


def setup_log_file(path: str):
    """打开日志文件用于本次运行追加写入。

    使用 UTF-8 编码，避免 Windows 默认 GBK 编码导致中文乱码。
    """
    global _log_file
    _log_file = open(path, "a", encoding="utf-8")


def close_log_file():
    """关闭日志文件句柄，若未打开则无操作。"""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def log(level: str, message: str, file_name: str = "", single: float = -1, total: float = -1):
    """统一格式的日志输出，同时写入控制台和日志文件。

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
        line = " ".join(parts)
        print(line)
        if _log_file is not None:
            _log_file.write(line + "\n")
            _log_file.flush()  # 即时落盘，防止异常退出丢失日志
