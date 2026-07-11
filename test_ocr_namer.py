import re
from unittest import TestCase

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
