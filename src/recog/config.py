from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class ModelSource(str, Enum):
	AUTO = "auto"
	LOCAL_PRETRAINED = "local_pretrained"
	LOCAL_FINETUNED = "local_finetuned"


MODEL_SOURCE: ModelSource = ModelSource.AUTO

# Optional overrides for local model roots when running inside this repo
LOCAL_PRETRAINED_ROOT: Optional[Path] = None
LOCAL_FINETUNED_ROOT: Optional[Path] = None


def resolve_model_settings(
	model_source: ModelSource,
	local_pretrained_root: Optional[Path],
	local_finetuned_root: Optional[Path],
) -> Tuple[str, Optional[Path]]:
	if model_source == ModelSource.LOCAL_PRETRAINED:
		root = local_pretrained_root or _default_pretrained_root()
		return "buffalo_l", root
	if model_source == ModelSource.LOCAL_FINETUNED:
		root = local_finetuned_root or _default_finetuned_root()
		return "uriel", root
	return "buffalo_l", None


def _default_pretrained_root() -> Path:
	repo_root = Path(__file__).resolve().parents[2]
	return repo_root / "models" / "pretrained"


def _default_finetuned_root() -> Path:
	repo_root = Path(__file__).resolve().parents[2]
	return repo_root / "models" / "finetuned"
