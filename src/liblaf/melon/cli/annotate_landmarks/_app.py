from pathlib import Path
from typing import Annotated

import typer


def cli(
    left_file: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    right_file: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    *,
    left_landmarks_file: Annotated[
        Path | None, typer.Option("-l", "--left-landmarks", dir_okay=False, exists=True)
    ] = None,
    right_landmarks_file: Annotated[
        Path | None,
        typer.Option("-r", "--right-landmarks", dir_okay=False, exists=True),
    ] = None,
) -> None:
    from ._main import main

    main(
        left_file,
        right_file,
        left_landmarks_file=left_landmarks_file,
        right_landmarks_file=right_landmarks_file,
    )
