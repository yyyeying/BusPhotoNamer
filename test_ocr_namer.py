import math
import re
from collections import Counter, defaultdict
from unittest import TestCase

from get_name_number import _fix_digit_misread
from ocr_namer import (aggregate_line_by_position, has_enough_results, line_weight,
                       penalize_missing_char_line, pick_winner)
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
        """LED 屏首位数字被误读为形近字母时应能还原。"""
        self.assertEqual(_fix_digit_misread("J31"), "331")
        self.assertEqual(_fix_digit_misread("S12"), "512")
        self.assertEqual(_fix_digit_misread("B18"), "818")

    def test_fix_digit_misread_no_fix(self):
        """数字部分已有 3 位、字母非形近、还原后非法线路号的情况不应还原。"""
        # 数字部分已完整，字母是屏边框噪声
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
        # 权重 = 最高分 × (1 + 0.5 × ln(次数))，仍远小于原先的线性累加
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
        """还原后的 3 位线路号应在投票中胜过车头碎片数字与背景线路号。"""
        # 各候选已按位置规则施加惩罚：与自编号同一行的车身碎片（×0.5），
        # 因首位字母还原而降权的截断候选（×0.6）。
        # 背景候选不施加水平惩罚：其偏移量与斜拍真线路号无法区分，故按原分参与投票
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
        self.assertEqual(pick_winner(weighted), "331",
                         "投票结果应为 331，实际权重：{}".format(dict(weighted)))

    def test_penalize_missing_char_line(self):
        """同一块 LED 屏读出长短两种读法时，漏字的较短读法应被降权。"""
        # 两个候选的 box 完全相同（同一处文本的两种读法），短读法是长读法的子序列
        led_box = (75.0, 36.0, 79.0, 38.0)
        line_list = [("93", 0.9550, led_box), ("963", 0.9477, led_box)]
        result, demoted = penalize_missing_char_line(line_list)
        scores = {text: score for text, score, _ in result}
        self.assertAlmostEqual(scores["93"], 0.9550 * 0.5)
        self.assertAlmostEqual(scores["963"], 0.9477, msg="较长的读法不应被降权")
        self.assertEqual(demoted, ["93<-963"])
        # 降权后 963 反超，最终投票应选 963
        weighted = {text: score for text, score, _ in result}
        self.assertEqual(pick_winner(weighted), "963")

    def test_penalize_missing_char_needs_same_position(self):
        """不同位置的候选即使构成子序列关系也不降权（可能是背景路牌的巧合）。"""
        line_list = [("32", 0.9000, (10.0, 10.0, 14.0, 12.0)),
                     ("362", 1.0000, (60.0, 40.0, 64.0, 42.0))]
        result, demoted = penalize_missing_char_line(line_list)
        scores = {text: score for text, score, _ in result}
        self.assertAlmostEqual(scores["32"], 0.9000)
        self.assertEqual(demoted, [])

    def test_penalize_missing_char_ignores_non_subsequence(self):
        """同位置但不构成子序列的候选不降权（如 '61' 与 '116'）。"""
        led_box = (60.0, 40.0, 64.0, 42.0)
        line_list = [("61", 0.9, led_box), ("116", 0.8, led_box)]
        result, demoted = penalize_missing_char_line(line_list)
        scores = {text: score for text, score, _ in result}
        self.assertAlmostEqual(scores["61"], 0.9)
        self.assertEqual(demoted, [])

    def test_pick_winner_is_deterministic(self):
        """权重相同时应稳定选出更长的取值，不受字典插入顺序影响。"""
        self.assertEqual(pick_winner({"93": 0.9, "963": 0.9}), "963")
        self.assertEqual(pick_winner({"963": 0.9, "93": 0.9}), "963")
        # 权重不同时仍按权重取胜者
        self.assertEqual(pick_winner({"93": 1.0, "963": 0.9}), "93")

    def test_has_enough_results_requires_lead(self):
        """线路号候选互相矛盾且领先不足时不应提前退出，需继续下一阶段取证。"""
        numbers = [("8645373", 1.0, (48.0, 61.0, 52.0, 63.0))] * 2
        plates = [("京A·42238D", 1.0, (75.0, 55.0, 79.0, 57.0))] * 2
        # 两个候选在不同位置，最高票领先不足 LINE_LEAD_RATIO
        conflicting = [("362", 1.0000, (60.0, 40.0, 64.0, 42.0)),
                       ("32", 0.9000, (10.0, 10.0, 14.0, 12.0))]
        self.assertFalse(has_enough_results(conflicting, numbers, plates),
                         "分歧过大时应继续取证")
        # 单一取值、或最高票明显领先时可以提前退出
        agreeing = [("963", 0.95, (75.0, 36.0, 79.0, 38.0)),
                    ("963", 0.99, (75.1, 36.1, 79.1, 38.1))]
        self.assertTrue(has_enough_results(agreeing, numbers, plates))

    def test_has_enough_results_requires_count(self):
        """任一字段候选不足 2 个时仍应继续取证。"""
        lines = [("963", 0.95, (75.0, 36.0, 79.0, 38.0))] * 2
        numbers = [("8645373", 1.0, (48.0, 61.0, 52.0, 63.0))] * 2
        self.assertFalse(has_enough_results(lines, numbers, []))


