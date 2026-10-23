"""Run project routes using one local TOML path configuration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from project_config import PROJECT_ROOT, ProjectPaths, load_paths


MIT_FILES = [
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
]

STAGES = {
    "quick": ["toy_random_forest.py"],
    "paper": ["step4_3f_official_split.py"],
    "extensions": [
        "step4_3f_official_split.py",
        "step5a_parse_nasa_v4.py",
        "step5b_zero_shot_v2.py",
        "step5c_decompose_v2.py",
        "step5d_survival_v2.py",
    ],
    "optimization": [
        "step4_3f_official_split.py",
        "step6a_main_optimization_v1.py",
        "step6b_transfer_optimization_v1.py",
        "step6c_survival_optimization_v1.py",
    ],
    "full": [
        "step1_capacity_fade.py",
        "step2_individual_plots.py",
        "step3_dqdv.py",
        "step4_mit_load.py",
        "step4_2_mit_features.py",
        "step4_3_rf_predict.py",
        "step4_3b_model_compare.py",
        "step4_3c_official_filter.py",
        "step4_3f_official_split.py",
        "step5a_parse_nasa_v4.py",
        "step5b_zero_shot_v2.py",
        "step5c_decompose_v2.py",
        "step5d_survival_v2.py",
        "step6a_main_optimization_v1.py",
        "step6b_transfer_optimization_v1.py",
        "step6c_survival_optimization_v1.py",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="quick")
    parser.add_argument("--config", type=Path, help="TOML path config; defaults to config.toml if present")
    parser.add_argument("--show-paths", action="store_true", help="print resolved data/cache/output paths")
    parser.add_argument("--dry-run", action="store_true", help="show scripts without running them")
    return parser.parse_args()


def validate_inputs(stage: str, paths: ProjectPaths) -> None:
    if stage == "quick":
        return
    missing_mit = [name for name in MIT_FILES if not (paths.mit_dir / name).is_file()]
    if missing_mit:
        raise FileNotFoundError(
            "MIT 数据缺失。按 data/README.md 下载并放入 "
            f"{paths.mit_dir}；缺少: {', '.join(missing_mit)}"
        )
    if stage in {"extensions", "full"}:
        nasa_files = list(paths.nasa_dir.glob("*.mat"))
        if len(nasa_files) != 34:
            raise FileNotFoundError(
                f"此路线需要 NASA 平铺 34 个 .mat 文件，当前目录 {paths.nasa_dir} 有 {len(nasa_files)} 个；"
                "请按 data/README.md 放置完整数据集。"
            )
    elif stage == "optimization":
        missing_nasa = [name for name in ("B0005.mat", "B0006.mat", "B0018.mat")
                        if not (paths.nasa_dir / name).is_file()]
        if missing_nasa:
            raise FileNotFoundError(
                f"step6 迁移实验需要 NASA 三颗标准电池，目录 {paths.nasa_dir} 缺少: "
                f"{', '.join(missing_nasa)}"
            )


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    # 子脚本继承 UTF-8 stdout：Windows 控制台默认是 cp1252/cp936，会让子脚本里的
    # 中文 print 崩掉（实测 GitHub windows-latest 上 `--stage quick` 红在
    # toy_random_forest.py 的中文/制表符输出）。project_config 的 import 侧已经有
    # 同一道守卫，这里再兜一层，覆盖不导入 project_config 的脚本；用户显式设置的
    # PYTHONIOENCODING 优先（setdefault 不覆盖）。
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if args.config:
        config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
        env["BATTERY_CONFIG"] = str(config_path.resolve())
    else:
        config_path = env.get("BATTERY_CONFIG")
    paths = load_paths(config_path, root=PROJECT_ROOT, environ=env)

    if args.show_paths:
        print(f"MIT data : {paths.mit_dir}")
        print(f"NASA data: {paths.nasa_dir}")
        print(f"Cache    : {paths.cache_dir}")
        print(f"Outputs  : {paths.output_dir}")
        return 0

    scripts = STAGES[args.stage]
    missing_scripts = [script for script in scripts if not (PROJECT_ROOT / "scripts" / script).is_file()]
    if missing_scripts:
        print(f"路线定义引用了不存在的脚本: {', '.join(missing_scripts)}", file=sys.stderr)
        return 2
    if not args.dry_run:
        try:
            validate_inputs(args.stage, paths)
        except FileNotFoundError as error:
            print(f"数据检查失败: {error}", file=sys.stderr)
            return 2
    print(f"路线 {args.stage}: " + " -> ".join(scripts))
    if args.dry_run:
        return 0

    for script in scripts:
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / script)]
        print(f"\n>>> {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        if completed.returncode:
            print(f"脚本失败，停止后续步骤: {script} (exit={completed.returncode})")
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())