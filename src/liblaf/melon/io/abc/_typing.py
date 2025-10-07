import functools
import types
from typing import Any

type RegType = type[Any] | types.UnionType
type SingleDispatchCallable[T] = functools._SingleDispatchCallable[T]  # noqa: SLF001
