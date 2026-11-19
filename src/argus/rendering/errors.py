"""Rendering errors. Messages carry counts and names, never patient text."""


class RenderingError(Exception):
    """Base for every rendering failure."""


class RenderingConfigError(RenderingError):
    """config/rendering.toml or the counter set is invalid."""


class RenderingBudgetError(RenderingError):
    """Mandatory content does not fit the total budget; no partial result."""
