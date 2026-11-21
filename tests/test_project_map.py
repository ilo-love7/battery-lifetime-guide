"""``docs/13_电池项目地图.md`` 与 README 关联表的同步性检查（完全离线）。

规则来自地图 §13.7：新增一个仓库进图，必须同时改三处 —— 地图 §13.2 表、
两个 README 的「关联仓库」表、以及本文件的 ``EXPECTED`` 清单。

这些测试不做网络访问，只读仓库内的 Markdown。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "docs" / "13_电池项目地图.md"

# 仓库名 → 层（与地图 §13.2 的「层」列一致）
EXPECTED = {
    "battery-cathode-ml": "材料",
    "pybamm-p2d-gsa": "器件",
    "battery-ecm-rul": "器件",
    "lithium-plating-detection": "器件",
    "battery-soc-ekf-ukf": "算法",
    "battery-soh-ev-online": "算法",
    "battery-fastcharge-opt": "算法",
    "battery-fault-diagnosis": "算法",
    "battery-thermal-management": "系统",
    "battery-soh-tsfm-benchmark": "基准",
}

LINK = re.compile(r"\[([A-Za-z0-9\-]+)\]\((https://github\.com/ilo-love7/([A-Za-z0-9\-]+))\)")


def _map_rows():
    """返回地图 §13.2 表的数据行（已按 ``|`` 切分并去掉首尾空列）。"""
    text = MAP.read_text(encoding="utf-8")
    start = text.index("## 13.2")
    end = text.index("## 13.3")
    rows = []
    for line in text[start:end].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or set(cells[0]) <= set("-: "):        # 分隔行
            continue
        if cells[0] in ("仓库", "Repository"):               # 表头
            continue
        rows.append(cells)
    return rows


def test_map_covers_expected_repositories_with_canonical_urls():
    """每个期望仓库出现且仅出现一次；链接名与仓库名一致。"""
    rows = _map_rows()
    names = []
    for cells in rows:
        m = LINK.search(cells[0])
        assert m, f"首列缺少 GitHub 链接: {cells[0]}"
        assert m.group(1) == m.group(3), f"链接文字与仓库名不一致: {cells[0]}"
        names.append(m.group(1))
    assert len(names) == len(set(names)), f"地图里有重复仓库: {names}"
    assert set(names) == set(EXPECTED), (
        f"地图与期望清单不一致：多 {sorted(set(names) - set(EXPECTED))}，"
        f"少 {sorted(set(EXPECTED) - set(names))}"
    )


def test_map_rows_are_complete_and_state_a_verifiable_number():
    """每行 5 列、层正确，且「可核验数字」列必须带数字（防止占位行）。"""
    rows = _map_rows()
    assert len(rows) == len(EXPECTED)
    for cells in rows:
        assert len(cells) == 5, f"列数不对（{len(cells)}）: {cells[0]}"
        name = LINK.search(cells[0]).group(1)
        assert EXPECTED[name] in cells[1], f"{name} 的层标注与期望不符: {cells[1]}"
        assert any(ch.isdigit() for ch in cells[3]), f"{name} 缺少可核验数字: {cells[3]}"


def test_both_readmes_link_every_repository_and_the_map():
    """两个 README 的关联表必须覆盖同一批仓库，并指向地图。"""
    for filename in ("README.md", "README.zh-CN.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        for name in EXPECTED:
            url = f"https://github.com/ilo-love7/{name}"
            assert url in text, f"{filename} 缺少仓库链接: {name}"
        assert "docs/13_电池项目地图.md" in text, f"{filename} 未指向 docs/13"


def test_map_is_linked_from_reading_paths():
    """方法论路线与 RELATED 都要能走到这张图。"""
    for filename in ("docs/00_怎么用这份攻略.md", "RELATED.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "13_电池项目地图.md" in text, f"{filename} 未指向 docs/13"
