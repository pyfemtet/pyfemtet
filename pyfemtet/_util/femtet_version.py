from packaging.version import Version

__all__ = [
    '_version'
]


def _version(
        main=None,
        major=None,
        minor=None,
        Femtet=None,
):
    if Femtet is not None:
        assert (main is None) and (major is None) and (minor is None), 'バージョンを指定しないでください'
        main, major, minor = Femtet.Version.split('.')[:3]
    else:
        assert (main is not None) and (major is not None) and (minor is not None), 'バージョンを指定してください'

    return Version('.'.join((str(main), str(major), str(minor))))
