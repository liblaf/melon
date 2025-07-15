from typing import Annotated

import cyclopts

from liblaf import grapes
from liblaf.melon import _version

from . import _annotate_landmarks, _info

app = cyclopts.App(name="melon", version=_version.__version__)


@app.meta.default
def init(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
) -> None:
    grapes.logging.init()
    app(tokens)


app.command(_annotate_landmarks.annotate_landmarks)
app.command(_info.info)
