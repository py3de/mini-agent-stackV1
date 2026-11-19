import pytest

from agent_stack.governance import AuditLog, set_audit

from agent_stack.hubspot_client import MockHubSpot, set_hubspot


@pytest.fixture()
def crm() -> MockHubSpot:
    """Fresh in-memory CRM for every test."""
    backend = MockHubSpot()
    set_hubspot(backend)
    return backend


@pytest.fixture(autouse=True)
def audit(tmp_path) -> AuditLog:
    """Every test writes to its own audit log file."""
    log = AuditLog(tmp_path / "audit.jsonl")
    set_audit(log)
    yield log
    set_audit(None)
