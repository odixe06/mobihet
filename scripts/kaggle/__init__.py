"""Shared Kaggle tools with compatibility imports for immutable notebook-building helpers."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2]/'.compat/scripts/kaggle'))
