from typing import Literal

import pydantic


def snake_to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class File(pydantic.BaseModel):
    name: str
    time: float


class Series(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(alias_generator=snake_to_kebab)
    file_series_version: Literal["1.0"] = "1.0"
    files: list[File] = []
