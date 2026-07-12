import logging

import pytest

from icle.log_setup import PACKAGE_LOGGER_NAME, enable_verbose_logging


@pytest.fixture(autouse=True)
def restore_icle_logger():
    """Undo any handler/level changes a test makes to the package logger."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    old_level = logger.level
    old_handlers = list(logger.handlers)
    yield
    for handler in list(logger.handlers):
        if handler not in old_handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(old_level)


def _verbose_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, "_icle_verbose_handler", False)]


def test_attaches_console_and_file_handlers(tmp_path):
    logger = enable_verbose_logging(str(tmp_path / "run.log"))

    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    console_handlers = [
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]

    assert file_handlers
    assert console_handlers
    assert logger.level == logging.DEBUG


def test_console_only_when_log_file_none():
    logger = enable_verbose_logging(None)

    assert not [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert _verbose_handlers(logger)


def test_reenabling_does_not_duplicate_handlers(tmp_path):
    enable_verbose_logging(str(tmp_path / "run.log"))
    logger = enable_verbose_logging(str(tmp_path / "run.log"))

    assert len(_verbose_handlers(logger)) == 2  # one console + one file


def test_step_messages_reach_log_file(tmp_path):
    log_file = tmp_path / "run.log"
    logger = enable_verbose_logging(str(log_file))

    # Component modules log via child loggers that propagate to "icle".
    logging.getLogger("icle.runtime.core").debug("Task [T1] output:\nhello")

    for handler in logger.handlers:
        handler.flush()
    assert "Task [T1] output" in log_file.read_text(encoding="utf-8")


def test_icle_constructor_enables_verbose(tmp_path, mock_model):
    from icle import ICLE

    log_file = tmp_path / "pipeline.log"
    ICLE(
        model=mock_model,
        global_task="Testing.",
        expert_save_dir=str(tmp_path),
        verbose=True,
        log_file=str(log_file),
    )

    assert log_file.exists()
    assert len(_verbose_handlers(logging.getLogger(PACKAGE_LOGGER_NAME))) == 2


def test_icle_constructor_defaults_to_quiet(tmp_path, mock_model):
    from icle import ICLE

    ICLE(
        model=mock_model,
        global_task="Testing.",
        expert_save_dir=str(tmp_path),
    )

    assert not _verbose_handlers(logging.getLogger(PACKAGE_LOGGER_NAME))
