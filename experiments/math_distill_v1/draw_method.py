"""Draw exact, editable Chinese method explainers as SVG and antialiased PNG."""
from __future__ import annotations

from html import escape
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).absolute().parents[2]
OUT = ROOT / "output/math_distill_v1/figures"
FONT = "/System/Library/Fonts/STHeiti Light.ttc"
BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
INK = "#15283C"
MUTED = "#52687A"
BLUE = "#2465CA"
TEAL = "#087F80"
ORANGE = "#AD6619"
PAPER = "#F4F7FB"
LINE = "#D5DFEA"


class Figure:
    def __init__(self, width, height):
        self.width, self.height, self.scale = width, height, 2
        self.image = Image.new("RGB", (width * self.scale, height * self.scale), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="white"/>']
        self.fonts = {}

    def font(self, size, bold=False):
        key = size, bold
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(BOLD if bold else FONT, size * self.scale)
        return self.fonts[key]

    def rect(self, x, y, w, h, fill="white", stroke=None, radius=22):
        box = tuple(int(v * self.scale) for v in (x, y, x + w, y + h))
        self.draw.rounded_rectangle(box, radius=radius * self.scale, fill=fill,
                                    outline=stroke, width=2 * self.scale)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="2"/>')

    def text(self, x, y, content, size=30, color=INK, bold=False, anchor="left"):
        font = self.font(size, bold)
        width = self.draw.textlength(content, font=font) / self.scale
        left = x - width / 2 if anchor == "center" else x
        self.draw.text((left * self.scale, y * self.scale), content, font=font, fill=color, anchor="lt")
        self.svg.append(f'<text x="{left:.2f}" y="{y + size * .88:.2f}" fill="{color}" font-size="{size}" font-family="Heiti SC, PingFang SC, Noto Sans CJK SC, sans-serif" font-weight="{700 if bold else 400}">{escape(content)}</text>')

    def paragraph(self, x, y, content, width, size=28, color=MUTED, leading=None, bold=False):
        leading = leading or int(size * 1.55)
        for source_line in content.split("\n"):
            line = ""
            for char in source_line:
                if self.draw.textlength(line + char, font=self.font(size, bold)) / self.scale > width and line:
                    self.text(x, y, line, size, color, bold)
                    y += leading
                    line = ""
                line += char
            if line:
                self.text(x, y, line, size, color, bold)
            y += leading
        return y

    def line(self, points, color=BLUE, width=3, arrow=True, dashed=False):
        scaled = [(x * self.scale, y * self.scale) for x, y in points]
        if dashed:
            for a, b in zip(points, points[1:]):
                length = math.dist(a, b)
                for d in range(0, math.ceil(length), 18):
                    end = min(d + 10, length)
                    p = (a[0] + (b[0] - a[0]) * d / length, a[1] + (b[1] - a[1]) * d / length)
                    q = (a[0] + (b[0] - a[0]) * end / length, a[1] + (b[1] - a[1]) * end / length)
                    self.draw.line([(p[0] * self.scale, p[1] * self.scale), (q[0] * self.scale, q[1] * self.scale)], fill=color, width=width * self.scale)
        else:
            self.draw.line(scaled, fill=color, width=width * self.scale)
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/>')
        if arrow:
            a, b = points[-2:]
            theta = math.atan2(b[1] - a[1], b[0] - a[0])
            shape = [b, (b[0] - 13 * math.cos(theta - .5), b[1] - 13 * math.sin(theta - .5)),
                     (b[0] - 13 * math.cos(theta + .5), b[1] - 13 * math.sin(theta + .5))]
            self.draw.polygon([(x * self.scale, y * self.scale) for x, y in shape], fill=color)
            self.svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x, y in shape)}" fill="{color}"/>')

    def chip(self, x, y, label, color=BLUE, fill="#EAF1FD", width=260):
        self.rect(x, y, width, 48, fill, radius=12)
        self.text(x + 18, y + 11, label, 25, color, True)

    def save(self, stem):
        OUT.mkdir(parents=True, exist_ok=True)
        png, svg = OUT / (stem + ".png"), OUT / (stem + ".svg")
        # Versioned files: refuse to overwrite unrelated generated figures.
        self.image.save(png, optimize=True)
        svg.write_text("\n".join(self.svg + ["</svg>"]) + "\n")
        print(png)
        print(svg)


def overview():
    f = Figure(2200, 1450)
    f.rect(0, 0, 2200, 228, PAPER, radius=0)
    f.text(64, 44, "轻量数学推理蒸馏 · 方法总览", 30, BLUE, True)
    f.text(64, 102, "用学生的困难与注意力，指导教师补哪里", 62, bold=True)
    f.text(64, 184, "训练时补充关键关系；推理时由 1.5B 学生独立解题。", 29, MUTED)
    f.text(66, 277, "01   离线诊断与训练", 31, bold=True)
    f.chip(1610, 264, "方法设计 · 效果待验证", ORANGE, "#FFF2DF", 525)
    cards = [
        ("1", "准备正确解答", "题目 + 教师原始 CoT", "逐条核查数学与完整性\n保留题目和答案来源\n固定训练题集合", BLUE, "#EEF4FF", "数据准备"),
        ("2", "找困难步骤", "学生预测难度 · NLL", "让学生读正确解答\n聚合每个步骤的难度\n各题使用自己的步骤边界", BLUE, "#EEF4FF", "定位候选"),
        ("3", "找相关前提", "学生 attention", "从预测位置回看前文\n寻找可能相关的条件\n候选关联需要语义核查", TEAL, "#E9F7F4", "读取功能已验证"),
        ("4", "补充局部解释", "教师说明“为何能这样推”", "联系前提、规则与当前步骤\n将解释放到困难步骤之前\n核查数学、长度与答案泄漏", ORANGE, "#FFF4E4", "局部改写待实施"),
        ("5", "训练轻量学生", "改写 CoT + 答案监督", "与基础蒸馏组公平比较\n固定模型与训练配置\n记录实际监督 token 数", ORANGE, "#FFF4E4", "本轮尚未训练"),
    ]
    for i, (number, title, subtitle, body, color, fill, status) in enumerate(cards):
        x = 64 + 420 * i
        f.rect(x, 340, 390, 421, fill, stroke=LINE)
        f.rect(x + 24, 365, 53, 53, color, radius=14)
        f.text(x + 50, 376, number, 30, "white", True, "center")
        f.text(x + 24, 445, title, 36, color, True)
        f.paragraph(x + 24, 503, subtitle, 342, 26, color, bold=True)
        f.paragraph(x + 24, 558, body, 344, 26, leading=43)
        f.text(x + 24, 713, status, 24, color, True)
        if i < 4:
            f.line([(x + 395, 548), (x + 415, 548)], color=MUTED, width=3)
    f.rect(64, 799, 2070, 74, "#F5F8FC", radius=16)
    f.text(88, 821, "对齐规则：预测某个目标 token 时，读取前一个 token 的查询；只看已有条件，不看未来。", 28, MUTED)
    f.text(66, 917, "02   训练后的独立推理", 31, bold=True)
    boxes = [(64, 1002, 460, "新题目", "正式测试题，不含训练提示"),
             (676, 1002, 680, "训练后的 1.5B 学生", "直接生成推导与最终答案"),
             (1508, 1002, 626, "独立评测", "最终答案准确率 + 错误类型 + 成本")]
    for x, y, w, title, sub in boxes:
        f.rect(x, y, w, 143, "#F0F5FC", stroke=LINE)
        f.text(x + w / 2, y + 29, title, 36, bold=True, anchor="center")
        f.text(x + w / 2, y + 89, sub, 25, MUTED, anchor="center")
    f.line([(537, 1073), (658, 1073)], color=BLUE, width=4)
    f.line([(1370, 1073), (1490, 1073)], color=BLUE, width=4)
    f.text(66, 1203, "怎样证明“更好”？", 32, bold=True)
    f.text(66, 1262, "对照：未训练学生 / 仅答案硬蒸馏 / 软蒸馏 / 原始 CoT / 我们的方法", 30, MUTED)
    f.text(66, 1311, "attention 的额外贡献：再与仅难度选位、随机或位置匹配控制比较。", 29, MUTED)
    f.line([(64, 1373), (2134, 1373)], color=LINE, width=2, arrow=False)
    f.text(66, 1400, "当前：开发基线运行中。attention 相关性不等于因果；完整方法的收益尚待实验。  |  2026-10-22", 25, MUTED)
    f.save("method_overview_zh_v1")


def example():
    f = Figure(2200, 1580)
    f.rect(0, 0, 2200, 226, PAPER, radius=0)
    f.text(64, 42, "一个具体例子 · 从题面条件到数学方程", 30, BLUE, True)
    f.text(64, 101, "attention 帮助寻找“这一步该联系什么”", 59, bold=True)
    f.text(64, 181, "概念示意：假设困难发生在建立周长方程处；这不是已测得的定位结论。", 29, MUTED)
    f.rect(64, 263, 2070, 124, "#F0F5FC", stroke=LINE)
    f.text(92, 286, "题目", 27, BLUE, True)
    f.text(92, 334, "矩形周长是 50 cm，长比宽多 5 cm。求矩形面积。", 35, bold=True)
    f.rect(64, 431, 793, 607, "white", stroke=LINE)
    f.text(96, 468, "A  正确解答中的推导步骤", 33, bold=True)
    f.text(96, 541, "设宽为 w，则长为 w + 5。", 31, MUTED)
    f.line([(440, 593), (440, 633)], width=4)
    f.rect(94, 659, 733, 158, "#FFF2DF", stroke="#E9C28A")
    f.text(119, 685, "若这一过渡的学生预测难度较高", 28, ORANGE, True)
    f.text(461, 748, "2w + 2(w + 5) = 50", 43, ORANGE, True, "center")
    f.line([(440, 834), (440, 872)], width=4)
    f.text(96, 909, "再解出宽与长，最后计算面积。", 31, MUTED)
    f.text(96, 979, "各题按自己的推导划分步骤，不固定成三步。", 25, MUTED)
    f.rect(946, 431, 1188, 607, "#F2FAF8", stroke="#CDE5DF")
    f.text(978, 468, "B  学生 attention 提供前提候选", 33, TEAL, True)
    f.paragraph(978, 535, "在预测困难目标 token 的位置之前，\n回看题干和已有推导；聚合到有意义的条件。", 1100, 30, leading=48)
    f.rect(978, 666, 530, 119, "white", stroke="#A6D4CA")
    f.text(1004, 690, "条件 A", 25, TEAL, True)
    f.text(1004, 738, "周长 = 50 cm", 33, bold=True)
    f.rect(1558, 666, 542, 119, "white", stroke="#A6D4CA")
    f.text(1585, 690, "条件 B", 25, TEAL, True)
    f.text(1585, 738, "长 = 宽 + 5 cm", 33, bold=True)
    f.line([(858, 742), (927, 742)], color=TEAL, width=4, dashed=True)
    f.text(978, 845, "这两条条件，是构造周长方程的候选依据。", 30, TEAL, True)
    f.paragraph(978, 907, "这里不使用虚构的 attention 热度值。实际最大权重可能落在问号或措辞上，因此还需要语义核查与消融对照。", 1110, 27, leading=43)
    f.line([(1530, 1048), (1530, 1102)], color=ORANGE, width=4)
    f.rect(64, 1117, 2070, 255, "#FFF5E7", stroke="#E9C28A")
    f.text(95, 1152, "C  教师补充“前提 → 规则 → 当前步骤”", 34, ORANGE, True)
    f.paragraph(95, 1216,
                "“矩形有两条长与两条宽，周长把四条边相加。先用‘长 = 宽 + 5’替换长，\n再把两条宽和两条长都计入周长。”",
                1970, 32, INK, leading=51)
    f.text(95, 1330, "将补充解释放在建立方程之前；核查数学与答案泄漏后，再用于改写 CoT 的蒸馏训练。", 27, ORANGE)
    f.text(66, 1422, "学生难度决定“哪里值得看”；attention 提供“可能联系什么”；教师负责“把关系解释清楚”。", 31, bold=True)
    f.line([(64, 1490), (2134, 1490)], color=LINE, width=2, arrow=False)
    f.text(66, 1518, "注意：这是待验证的方法设计。该例未展示实测注意力排名，也不声称定位到了真正的理解缺口。", 26, MUTED)
    f.save("method_example_rectangle_zh_v1")


if __name__ == "__main__":
    overview()
    example()
