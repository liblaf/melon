import importlib.metadata
from typing import Annotated

import cyclopts
import liblaf.logging


def _get_version() -> str:
    return importlib.metadata.version("liblaf-melon")


app: cyclopts.App = cyclopts.App("melon", version=_get_version)
app.register_install_completion_command(add_to_startup=False)
app.command("liblaf.melon.cli:annotate_landmarks")


@app.meta.default
def _(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
) -> None:
    liblaf.logging.init()
    app(tokens)
