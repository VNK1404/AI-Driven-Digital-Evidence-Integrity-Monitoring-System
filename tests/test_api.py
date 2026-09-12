"""
tests/test_api.py
------------------
Tests for api/app.py routes and UI endpoint rendering.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_ui_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Forensic Workbench" in res.data
    assert b"AI-Driven Evidence Integrity System" in res.data


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "report_count" in data


def test_reports_list(client):
    res = client.get("/reports")
    assert res.status_code == 200
    data = res.get_json()
    assert "evidence_ids" in data
