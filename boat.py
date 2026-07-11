from paddleocr import PaddleOCR

from singleton import singleton


@singleton
class Boat:
    def __init__(self):
        """持有 PaddleOCR 实例，省得每次都初始化。因为 Paddle 是桨，那当然要配船了！"""
        try:
            self.paddle: PaddleOCR = PaddleOCR(use_angle_cls=True, use_gpu=True)
        except Exception:
            # GPU 不可用时降级到 CPU
            self.paddle: PaddleOCR = PaddleOCR(use_angle_cls=True, use_gpu=False)
