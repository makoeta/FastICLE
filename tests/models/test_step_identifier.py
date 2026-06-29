from icle.models.step_identifier import STEP_IDENTIFIER


def test_dispatch_value():
    assert STEP_IDENTIFIER.DISPATCH.value == "dispatch"


def test_cast_value():
    assert STEP_IDENTIFIER.CAST.value == "cast"


def test_runtime_value():
    assert STEP_IDENTIFIER.RUNTIME.value == "runtime"


def test_assemble_value():
    assert STEP_IDENTIFIER.ASSEMBLE.value == "assemble"


def test_all_four_members():
    assert len(STEP_IDENTIFIER) == 4


def test_all_values():
    values = {e.value for e in STEP_IDENTIFIER}
    assert values == {"dispatch", "cast", "runtime", "assemble"}


def test_members_accessible_by_name():
    assert STEP_IDENTIFIER["DISPATCH"] is STEP_IDENTIFIER.DISPATCH
    assert STEP_IDENTIFIER["CAST"] is STEP_IDENTIFIER.CAST
    assert STEP_IDENTIFIER["RUNTIME"] is STEP_IDENTIFIER.RUNTIME
    assert STEP_IDENTIFIER["ASSEMBLE"] is STEP_IDENTIFIER.ASSEMBLE
