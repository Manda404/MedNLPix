# src/mednlpix/pipelines/registry_manager.py

import json
import joblib
from pathlib import Path
from mednlpix.utils.path_utils import find_project_root, get_prefix_path



def update_registry(model_path: str | Path, model_type: str, registry_path: str = "models/registry.json") -> None:
    """
    Update the registry.json file with the latest version of a pipeline or model.
    """
    model_path = Path(model_path).resolve()
    prefix = get_prefix_path(model_path, stop_dir="models")

    # Build the full path to the registry file
    full_registry_path = prefix / registry_path
    full_registry_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Load existing registry if it exists
    if full_registry_path.exists():
        with open(full_registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    # Update or create entry
    registry[model_type] = {
        "path": str(model_path),
        "version": model_path.stem,
    }

    # Save registry file
    with open(full_registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    print(f"Registry updated for '{model_type}': {model_path}")


def load_model_from_registry(model_type: str, registry_rel_path: str = "src/mednlpix/models/registry.json"):
    """
    Load a model or pipeline registered under a specific model_type.
    Automatically resolves the full path of the registry file from the project root.
    """
    project_root = find_project_root()
    registry_file = project_root / registry_rel_path

    if not registry_file.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_file}")

    with open(registry_file, "r") as f:
        registry = json.load(f)

    if model_type not in registry:
        raise KeyError(f"Model type '{model_type}' not found in registry.")

    model_path = Path(registry[model_type]["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at path: {model_path}")

    print(f"Loading model from {model_path}")
    return joblib.load(model_path)