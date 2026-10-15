from pathlib import Path

from scripts.project_config import load_paths


def test_default_paths_are_repo_relative(tmp_path: Path):
    paths = load_paths(root=tmp_path, environ={})
    assert paths.mit_dir == tmp_path / "data" / "mit"
    assert paths.nasa_dir == tmp_path / "data" / "nasa"
    assert paths.cache_dir == tmp_path / ".cache"
    assert paths.output_dir == tmp_path / "results" / "generated"


def test_toml_paths_resolve_relative_to_project_root(tmp_path: Path):
    config = tmp_path / "local.toml"
    config.write_text(
        '[paths]\nmit_dir = "datasets/mit"\nnasa_dir = "../nasa"\n',
        encoding="utf-8",
    )
    paths = load_paths(config, root=tmp_path, environ={})
    assert paths.mit_dir == tmp_path / "datasets" / "mit"
    assert paths.nasa_dir == tmp_path.parent / "nasa"


def test_environment_overrides_toml(tmp_path: Path):
    config = tmp_path / "local.toml"
    config.write_text('[paths]\nmit_dir = "datasets/mit"\n', encoding="utf-8")
    external_mit_dir = tmp_path.parent / "external-mit"
    paths = load_paths(
        config,
        root=tmp_path,
        environ={"BATTERY_MIT_DIR": str(external_mit_dir)},
    )
    assert paths.mit_dir == external_mit_dir