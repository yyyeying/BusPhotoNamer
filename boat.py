from paddleocr import PaddleOCR

from singleton import singleton


@singleton
class Boat:
    def __init__(self):
        """持有 PaddleOCR 实例，省得每次都初始化。因为 Paddle 是桨，那当然要配船了！"""
        self.paddle: PaddleOCR = PaddleOCR(
            use_textline_orientation=False,
            text_rec_score_thresh=0.75,
        )
