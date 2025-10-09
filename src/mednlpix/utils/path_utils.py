from pathlib import Path

def find_project_root() -> Path:
    """
    Automatically detect the project root (where pyproject.toml or src/ exists).
    """
    current_dir = Path().resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "src").is_dir():
            return parent
    return current_dir


def get_prefix_path(full_path: str | Path, stop_dir: str = "models") -> Path:
    """
    Return the prefix (base) path of a full path before the given directory name.

    Example:
        Input:
            /Users/.../mednlpix/models/pipeline.joblib
        Output:
            /Users/.../mednlpix/
    """
    path = Path(full_path).resolve()
    parts = path.parts

    # Find the index of the stop directory
    if stop_dir in parts:
        stop_index = parts.index(stop_dir)
        prefix = Path(*parts[:stop_index])
        return prefix
    else:
        raise ValueError(f"'{stop_dir}' not found in path: {full_path}")