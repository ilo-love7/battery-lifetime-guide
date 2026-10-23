"""Windows 控制台编码回归测试：CLI 在 cp1252 的 stdout 下也必须 exit 0。

为什么值得一个测试：本仓库的脚本会打印中文与制表/箭头字符（决策树里的
``├─ 是 →``），而 Windows 控制台默认编码是 cp1252 / cp936。2026-10-23 给 CI 加
``windows-latest`` leg 时，``python scripts/run_project.py --stage quick`` 直接崩在
``print(f"路线 ...")``：

    UnicodeEncodeError: 'charmap' codec can't encode characters ...

修法是 ``scripts/project_config.enable_utf8_stdout()``（模块级调用，覆盖 scripts/ 下
19 个会打印非 ASCII 的脚本）+ ``run_project.py`` 给子进程再兜一层
``PYTHONIOENCODING=utf-8``。本文件把这条性质钉住，而不是靠人记得"Windows 要设个环境变量"。

边界——这个文件**不能**证明什么：只断言"命令不崩 + 中文确实写出去了"。字符集是否完整
不在覆盖范围：守卫用 ``errors="replace"`` 兜底，个别字符被打成 ``?`` 仍算通过。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# 模拟 GitHub windows-latest runner 的默认控制台：cp1252，且不做 UTF-8 模式。
NARROW_CONSOLE_ENV = {
    "PYTHONIOENCODING": "cp1252",
    "PYTHONUTF8": "0",
    "MPLBACKEND": "Agg",  # runner 没有显示设备，别让 matplotlib 去开 Tk 窗口
}


def _run(args) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(NARROW_CONSOLE_ENV)
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_guard_reconfigures_stdout_to_utf8_under_cp1252():
    """导入 project_config 之后 stdout 必须是 UTF-8——这是所有 19 个脚本的兜底。"""
    code = (
        "import sys; sys.path.insert(0, 'scripts'); import project_config; "
        "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8', "
        "f'stdout still {sys.stdout.encoding}'; print('guard active')"
    )
    proc = _run(["-c", code])
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "guard active" in proc.stdout


def test_full_route_dry_run_survives_cp1252_console():
    """CI 会跑的第一步：打印路线映射（含中文），必须不崩。"""
    proc = _run([str(SCRIPTS / "run_project.py"), "--stage", "full", "--dry-run"])
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "路线" in proc.stdout, (
        "中文路线名应当真的写出去（而不是崩掉或被替换光）"
    )


def test_quick_route_survives_cp1252_console():
    """CI 会跑的第二步：`--stage quick` 本地实跑，并会 spawn toy_random_forest.py。"""
    proc = _run([str(SCRIPTS / "run_project.py"), "--stage", "quick"])
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"


def test_toy_random_forest_direct_survives_cp1252_console():
    """直接调用子脚本（不经 run_project）也必须能打印制表符与箭头。"""
    proc = _run([str(SCRIPTS / "toy_random_forest.py")])
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "├─" in proc.stdout or "是" in proc.stdout, (
        "决策树的分支字符/中文应当写出去；若这里空了说明守卫没生效"
    )


def test_protocol_drift_evidence_survives_cp1252_console():
    """README 让用户在本地跑这条命令取证据，Windows 上同样不得崩。"""
    proc = _run([str(SCRIPTS / "protocol_drift.py"), "--evidence"])
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "committed-artifact evidence" in proc.stdout