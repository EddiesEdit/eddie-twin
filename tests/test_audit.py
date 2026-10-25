
import pytest

from eddie_twin.security.audit import SecurityAudit


def test_audit_records_security_event():
    audit = SecurityAudit()

    audit.record(
        "input_blocked",
        "instruction_override",
        blocked=True,
    )

    report = audit.report()

    assert report["total_events"] == 1
    assert report["total_blocked"] == 1
    assert report["recent_events"][0]["blocked"] is True


def test_audit_does_not_store_raw_message():
    audit = SecurityAudit()

    audit.record("input_blocked", "injection", blocked=True)

    assert "input" not in audit.report()["recent_events"][0]


def test_audit_retains_only_latest_events():
    audit = SecurityAudit(max_events=2)

    audit.record("event_1", "test", blocked=False)
    audit.record("event_2", "test", blocked=True)
    audit.record("event_3", "test", blocked=False)

    report = audit.report()

    assert report["total_events"] == 2
    assert report["total_blocked"] == 1


def test_audit_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="positive"):
        SecurityAudit(max_events=0)
