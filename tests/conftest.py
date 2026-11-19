import pytest

from agent_stack.hubspot_client import MockHubSpot, set_hubspot


@pytest.fixture()
def crm() -> MockHubSpot:
    """Fresh in-memory CRM for every test."""
    backend = MockHubSpot()
    set_hubspot(backend)
    return backend
