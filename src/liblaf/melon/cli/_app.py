from typing import Annotated

import cyclopts

from liblaf import grapes
from liblaf.melon import _version

from . import _info

app = cyclopts.App(name="melon", version=_version.__version__)


@app.meta.default
def init(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
) -> None:
    grapes.init_logging()
    app(tokens)


app.command(_info.info)


def main() -> None:
    app.meta()
