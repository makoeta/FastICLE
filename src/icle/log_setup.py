import logging
import sys

PACKAGE_LOGGER_NAME = "icle"

# Attribute set on handlers created here so re-enabling replaces them
# instead of stacking duplicates.
_HANDLER_MARKER = "_icle_verbose_handler"

_FORMAT = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def enable_verbose_logging(log_file: str | None = "icle.log") -> logging.Logger:
    """Log every pipeline step of the `icle` package to the console and,
    unless `log_file` is None, to a file.

    Console output goes to stderr. Safe to call repeatedly: previously
    attached verbose handlers are replaced, not duplicated.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    for handler in [h for h in logger.handlers if getattr(h, _HANDLER_MARKER, False)]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    setattr(console_handler, _HANDLER_MARKER, True)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, True)
        logger.addHandler(file_handler)

    return logger
