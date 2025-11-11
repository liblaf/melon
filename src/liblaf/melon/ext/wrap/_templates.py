import jinja2
import wrapt


def _factory() -> jinja2.Environment:
    return jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        autoescape=jinja2.select_autoescape(),
        loader=jinja2.PackageLoader("liblaf.melon.ext.wrap"),
    )


environment: jinja2.Environment = wrapt.LazyObjectProxy(_factory)  # pyright: ignore[reportAssignmentType]
