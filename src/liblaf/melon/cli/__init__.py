"""Command-line interface entry points."""

from ._annotate_landmarks import annotate_landmarks
from ._app import app

__all__ = ["annotate_landmarks", "app"]
