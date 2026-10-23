"""Shared local path configuration for the standalone teaching scripts."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def enable_utf8_stdout() -> None:
    """把 stdout / stderr 切到 UTF-8，让中文与制表/箭头字符在 Windows 上也能打印。

    为什么需要：本仓库的脚本会打印中文与制表符（例如决策树里的 ``├─ 是 →``），
    而 Windows 控制台默认编码是 cp1252 / cp936。实测 GitHub 的 windows-latest
    runner 上 ``python scripts/run_project.py --stage quick`` 直接崩在
    ``print(f"路线 ...")``：

        UnicodeEncodeError: 'charmap' codec can't encode characters ...

    ``errors="replace"`` 是兜底：万一某个字符连 UTF-8 都写不出去，宁可打成 ``?``，
    也不要让整条路线在最后一步倒下。

    在本模块**模块级**调用是有意的：``scripts/`` 下会打印非 ASCII 的脚本里绝大多数
    都 ``from project_config import ...``，所以这一处就覆盖了 CI 会跑的整条路线；
    剩下三个不导入本模块的脚本（protocol_drift / render_github_banner /
    step5d_validate_km）自行显式调用。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # 流被换成非 TextIOWrapper（例如 pytest 的捕获对象）或已关闭：无需处理。
            pass


enable_utf8_stdout()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULTS = {
    "mit_dir": "data/mit",
    "nasa_dir": "data/nasa",
    "cache_dir": ".cache",
    "output_dir": "results/generated",
}
_ENV_KEYS = {
    "mit_dir": "BATTERY_MIT_DIR",
    "nasa_dir": "BATTERY_NASA_DIR",
    "cache_dir": "BATTERY_CACHE_DIR",
    "output_dir": "BATTERY_OUTPUT_DIR",
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    mit_dir: Path
    nasa_dir: Path
    cache_dir: Path
    output_dir: Path


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_paths(
    config_path: str | Path | None = None,
    *,
    root: str | Path = PROJECT_ROOT,
    environ: Mapping[str, str] | None = None,
) -> ProjectPaths:
    """Resolve TOML settings, then environment overrides, then safe defaults."""
    root_path = Path(root).expanduser().resolve()
    env = os.environ if environ is None else environ
    configured_file = config_path or env.get("BATTERY_CONFIG")
    if configured_file is None:
        candidate = root_path / "config.toml"
        configured_file = candidate if candidate.is_file() else None

    values = dict(_DEFAULTS)
    if configured_file is not None:
        config_file = Path(configured_file).expanduser()
        if not config_file.is_absolute():
            config_file = root_path / config_file
        with config_file.open("rb") as source:
            config = tomllib.load(source)
        paths = config.get("paths", {})
        for key in values:
            if key in paths:
                values[key] = paths[key]

    for key, env_key in _ENV_KEYS.items():
        if env.get(env_key):
            values[key] = env[env_key]

    return ProjectPaths(
        root=root_path,
        mit_dir=_resolve_path(values["mit_dir"], root_path),
        nasa_dir=_resolve_path(values["nasa_dir"], root_path),
        cache_dir=_resolve_path(values["cache_dir"], root_path),
        output_dir=_resolve_path(values["output_dir"], root_path),
    )


PATHS = load_paths()