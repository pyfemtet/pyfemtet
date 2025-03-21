__all__ = [
    'float_',
]


def float_(value: str | None | float) -> str | float | None:
    if value is None:
        return value
    try:
        return float(value)
    except ValueError:
        return value
