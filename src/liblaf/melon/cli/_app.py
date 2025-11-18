from typing import Annotated

import cyclopts

from liblaf import grapes
from liblaf.melon._version import __version__

app = cyclopts.App(name="melon", version=__version__)
app.register_install_completion_command(add_to_startup=False)
app.command("liblaf.melon.cli:annotate_landmarks")
app.command("liblaf.melon.cli:info")


@app.meta.default
def init(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
) -> None:
    grapes.logging.init()
    app(tokens)
