import typer

from liblaf import grapes

from . import annotate_landmarks

app = typer.Typer(name="melon")
app.command(name="annotate-landmarks")(annotate_landmarks.cli)


@app.callback()
def callback() -> None:
    grapes.init_logging()
