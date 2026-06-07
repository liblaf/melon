import functools

import jinja2 as j2


@functools.cache
def get_environment() -> j2.Environment:
    """Return the cached Jinja environment for Wrap project templates."""
    return j2.Environment(
        undefined=j2.StrictUndefined,
        autoescape=j2.select_autoescape(),
        loader=j2.PackageLoader("liblaf.melon.ext.wrap"),
    )
