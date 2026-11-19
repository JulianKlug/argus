"""Window rendering: DecisionWindow -> deterministic clinical prompt text within a token budget.

Spec: docs/spec_rendering.md.
"""

from argus.rendering.config import RenderingConfig, load_config
from argus.rendering.errors import RenderingBudgetError, RenderingConfigError, RenderingError
from argus.rendering.renderer import Renderer
from argus.rendering.types import EvidenceRef, RefKind, RenderedWindow, Section, SourceRow

__all__ = [
    "EvidenceRef",
    "RefKind",
    "RenderedWindow",
    "Renderer",
    "RenderingBudgetError",
    "RenderingConfig",
    "RenderingConfigError",
    "RenderingError",
    "Section",
    "SourceRow",
    "load_config",
]
