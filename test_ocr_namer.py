import math
import re
from collections import Counter, defaultdict
from unittest import TestCase

from get_name_number import _fix_digit_misread
from ocr_namer import aggregate_line_by_position, line_weight
from regex import bpt_line_regex, non_bpt_line_regex, number_regex, id_regex




class Test(TestCase):
    def test_bpt_line_regex(self):
        """测试公交集团线路号正则。"""
        should_match = ["345", "345快", "专12", "BRT1", "夜21", "快速直达专线15",
                        "F1", "C101", "Y9", "T2", "定制公交"]
        for text in should_match:
            result = re.findall(bpt_line_regex, text)
            self.assertEqual(len(result), 1, "应匹配线路号：{}，实际：{}".format(text, result))

    def test_bpt_line_regex_no_match(self):
        """测试不应匹配公交集团线路号的字符串。"""
        should_not_match = ["顺22", "昌1", "兴5", "密3", "京A·40871F", "834815"]
        for text in should_not_match:
            result = re.findall(bpt_line_regex, text)
            self.assertNotIn(text, result, "不应匹配为线路号：{}".format(text))

    def test_non_bpt_line_regex(self):
        """测试非公交集团郊县线路号正则。"""
        should_match = ["昌1", "顺22", "兴5", "空港1", "郊89", "密3"]
        for text in should_match:
            result = re.match(non_bpt_line_regex, text)
            self.assertIsNotNone(result, "应匹配郊县线路号：{}".format(text))

    def test_number_regex(self):
        """测试车辆自编号正则。"""
        should_match = ["1834815", "D834100", "B123456", "兴-01-2345"]
        for text in should_match:
            result = re.findall(number_regex, text)
            self.assertEqual(len(result), 1, "应匹配自编号：{}，实际：{}".format(text, result))

    def test_id_regex(self):
        """测试车牌号正则。"""
        should_match = ["京A·40871F", "京A·AS236"]
        for text in should_match:
            result = re.findall(id_regex, text)
            self.assertEqual(len(result), 1, "应匹配车牌号：{}，实际：{}".format(text, result))

    def test_jiao_not_in_bpt(self):
        """郊县线路号前缀不应被 bpt_line_regex 匹配为完整线路号。"""
        result = re.findall(bpt_line_regex, "郊89")
        self.assertNotIn("郊89", result, "郊89 不应作为完整线路号匹配公交集团正则")

    def test_xing_number_pattern(self):
        """兴-XX-XXXX 格式的自编号应能被正确匹配（之前因缺少 | 分隔符而失败）。"""
        result = re.findall(number_regex, "兴-01-2345")
        self.assertEqual(len(result), 1, "应匹配兴-01-2345，实际：{}".format(result))
        self.assertEqual(result[0], "兴-01-2345")

    def test_ye_line_pattern(self):
        """夜班线路号应能被正确匹配（之前因多余的 ] 而失败）。"""
        result = re.findall(bpt_line_regex, "夜21")
        self.assertEqual(len(result), 1, "应匹配夜21，实际：{}".format(result))
        self.assertEqual(result[0], "夜21")

    def test_fix_digit_misread(self):
        """LED 屏首位数字被误读为形近字母时应能还原（实测 331 被读成 J31）。"""
        self.assertEqual(_fix_digit_misread("J31"), "331")
        self.assertEqual(_fix_digit_misread("S12"), "512")
        self.assertEqual(_fix_digit_misread("B18"), "818")

    def test_fix_digit_misread_no_fix(self):
        """数字部分已有 3 位、字母非形近、还原后非法线路号的情况不应还原。"""
        # A447/H403 的字母是屏边框噪声，数字部分已完整
        self.assertIsNone(_fix_digit_misread("A447"))
        self.assertIsNone(_fix_digit_misread("H403"))
        # X 不在形近字母表中
        self.assertIsNone(_fix_digit_misread("X31"))
        # O 还原为 0，"031" 不是合法线路号
        self.assertIsNone(_fix_digit_misread("O31"))
        # 纯数字无需还原
        self.assertIsNone(_fix_digit_misread("331"))

    def test_aggregate_line_by_position(self):
        """同一位置被多个图像变体重复识别的线路号应聚合为一条，并按次数对数加成。"""
        # 车头碎片 '5' 在同一位置被 4 个变体读出，真实线路号在另一位置被读出 2 次
        line_list = [("5", 0.9541, (55.0, 58.0, 57.0, 60.0)),
                     ("5", 0.8724, (55.1, 58.1, 57.1, 60.1)),
                     ("5", 0.8174, (55.2, 58.2, 57.2, 60.2)),
                     ("5", 0.9539, (55.0, 58.0, 57.0, 60.0)),
                     ("331", 0.7586, (60.0, 44.0, 64.0, 46.0)),
                     ("331", 0.7945, (60.1, 44.1, 64.1, 46.1))]
        result = aggregate_line_by_position(line_list)
        scores = {text: score for text, score, _ in result}
        self.assertEqual(len(result), 2, "同位置重复项应各自聚合为一条，实际：{}".format(result))
        # 权重 = 最高分 × (1 + 0.5 × ln(次数))，仍远小于原先的线性累加（3.60 / 1.55）
        self.assertAlmostEqual(scores["5"], 0.9541 * (1 + 0.5 * math.log(4)))
        self.assertAlmostEqual(scores["331"], 0.7945 * (1 + 0.5 * math.log(2)))
        self.assertLess(scores["5"], 0.9541 * 4, "加成必须是次线性的")

    def test_dedup_line_keeps_other_positions(self):
        """同一取值出现在不同位置（如背景站牌）时应各自保留计票。"""
        line_list = [("671", 0.9, (10.0, 10.0, 14.0, 12.0)),
                     ("671", 0.8, (60.0, 40.0, 64.0, 42.0)),
                     ("671", 0.7, None)]
        result = aggregate_line_by_position(line_list)
        self.assertEqual(len(result), 3, "不同位置及无位置信息的候选都应保留")


    def test_line_weight_single_digit_penalty(self):
        """1 位纯数字线路号应降权，多位数与含汉字线路号不受影响。"""
        self.assertLess(line_weight("5", 1.0), 1.0)
        self.assertEqual(line_weight("331", 1.0), 1.0)
        self.assertEqual(line_weight("夜21", 1.0), 1.0)

    def test_vote_prefers_restored_line(self):
        """复现 ADSC04938：还原后的 331 应在投票中胜过车头碎片 5 与背景 111。"""
        # 各候选的置信度取自日志，并已按新规则施加位置惩罚：
        # 车头碎片 '5' 与自编号同一行（×0.5），被截断的 '31' 因首位字母还原而降权（×0.6）。
        # 背景 '111' 不施加水平惩罚：其偏移量与斜拍真线路号无法区分，故按原分参与投票
        line_list = [("331", 0.7586, (60.0, 44.0, 64.0, 46.0)),
                     ("31", 0.7586 * 0.6, (60.0, 44.0, 64.0, 46.0)),
                     ("331", 0.7945, (60.1, 44.1, 64.1, 46.1)),
                     ("31", 0.7945 * 0.6, (60.1, 44.1, 64.1, 46.1)),
                     ("5", 0.9541 * 0.5, (55.0, 58.0, 57.0, 60.0)),
                     ("5", 0.8724 * 0.5, (55.1, 58.1, 57.1, 60.1)),
                     ("5", 0.8174 * 0.5, (55.2, 58.2, 57.2, 60.2)),
                     ("5", 0.9539 * 0.5, (55.0, 58.0, 57.0, 60.0)),
                     ("111", 0.8663, (81.0, 49.0, 85.0, 51.0))]

        weighted = defaultdict(float)
        counts = Counter()
        for text, score, _ in aggregate_line_by_position(line_list):
            weighted[text] += line_weight(text, score)
            counts[text] += 1
        self.assertEqual(max(weighted, key=weighted.get), "331",
                         "投票结果应为 331，实际权重：{}".format(dict(weighted)))

