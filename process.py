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
