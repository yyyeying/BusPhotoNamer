# BusPhotoNamer

公交车照片 OCR 自动重命名工具。通过 PaddleOCR 识别照片中的线路号、车辆自编号、车牌号，自动重命名文件。

## 命名格式

```
{线路号}路{自编号}_{车牌号}.jpg          # 公交集团线路
{线路号}路{车牌号}.jpg                    # 郊县线路（无自编号）
{线路号}路{自编号}_{车牌号}_{原文件名}.jpg  # 存在 unknown 时附加原名
```

示例：`666路2632924_京A·40871F.jpg`

## 安装

### 1. 创建 conda 环境（Python 3.10）

```bash
conda create -n paddle3 python=3.10 -y
conda activate paddle3
```

### 2. 安装 PaddlePaddle GPU

```bash
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

### 3. 安装 PaddleOCR 及其他依赖

```bash
pip install paddleocr Pillow
```

## 配置

创建 `config_local.py` 设置默认照片目录（不提交到 GitHub）：

```python
DEFAULT_IMAGE_PATH = r"D:\你的照片目录"
```

## 使用

```bash
# 使用默认目录
python main.py

# 指定目录
python main.py "D:\照片\北京公交"
```

程序会递归遍历子目录下所有图片（`.jpg` / `.jpeg` / `.png` / `.bmp` / `.webp`）。

## 工作流程

1. **跳过已识别文件** — 文件名已包含完整识别结果且线路号 ≥ 2 位的跳过
2. **重新识别** — 含 `unknown` 字段或线路号为 1 位数的文件重新处理
3. **三阶段递进 OCR**
   - Phase 1：3 张基础图像（原图 / 模糊锐化 / 缩放锐化）
   - Phase 2：9 张 RGB 通道拆分（仅在 Phase 1 不足时）
   - Phase 3：9 张 Otsu 自适应二值化（仅在 Phase 2 不足时）
4. **确定性去重** — 线路号是自编号/车牌号的子串时按出现次数决定是否删除
5. **置信度加权投票** — 累计 OCR 置信度，取最高者作为最终结果
6. **重命名** — 按命名格式重命名；全 unknown 时跳过

## 识别范围

| 类型 | 正则 | 示例 |
|------|------|------|
| 公交集团线路号 | `bpt_line_regex` | `345`, `345快`, `专12`, `特5`, `特5快`, `运通101`, `BRT1`, `夜21`, `快速直达专线15` |
| 郊县线路号 | `non_bpt_line_regex` | `昌1`, `顺22`, `兴5`, `空港1`, `郊89`, `密3` |
| 车辆自编号 | `number_regex` | `1234`, `1834815`, `D834100`, `B123456`, `兴-01-2345` |
| 车牌号 | `id_regex` | `京A·40871F` |

## 项目结构

```
BusPhotoNamer/
├── main.py              # 入口，递归遍历目录
├── ocr_namer.py         # 核心逻辑：变体生成、去重、投票、重命名
├── get_name_number.py   # OCR 调用 + 正则分类
├── regex.py             # 四套正则模式
├── boat.py              # PaddleOCR 单例封装
├── process.py           # 进度跟踪 + 统一日志
├── singleton.py         # 线程安全单例装饰器
├── test_ocr_namer.py    # 正则单元测试
├── requirements.txt     # 依赖
├── config_local.py      # 本地配置（不提交）
└── example/             # 示例照片
```

## 运行测试

```bash
python -m unittest test_ocr_namer -v
```

## 环境要求

- NVIDIA GPU（≥ 8GB VRAM）
- CUDA 12.9 驱动
- Python 3.10+
- PaddlePaddle 3.0+ / PaddleOCR 3.0+

## License

MIT
