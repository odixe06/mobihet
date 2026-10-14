"""Shared tools; historical method-tool imports resolve through the hidden compatibility map."""
from pathlib import Path
import sys

legacy=str(Path(__file__).resolve().parents[1]/'.compat/scripts')
__path__.append(legacy)
if legacy not in sys.path:sys.path.append(legacy)
