"""
Unit and Integration Tests for the Responsible AI & Fair-Play Layer
===================================================================
Verifies:
- SHA-256 Cryptographic commitment scheme (zero-peeking assurance).
- Data consent controls and Right-to-be-Forgotten data purging.
- Explainability card generation.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from responsible_ai import (
    CryptographicCommitment,
    DataConsentManager,
    format_explainability_card,
)


def test_cryptographic_commitment_zero_peeking():
    """Verify cryptographic commitment creation and verification."""
    comm = CryptographicCommitment()

    # Agent commits to 'R'
    comm_hash = comm.create_commitment("R")
    assert len(comm_hash) == 64  # SHA-256 hex string

    # Reveal commitment
    revealed_move, salt, stored_hash = comm.reveal()
    assert revealed_move == "R"
    assert stored_hash == comm_hash

    # Verification must succeed with correct move & salt
    assert CryptographicCommitment.verify("R", salt, comm_hash) is True

    # Verification must fail if move is altered (tamper evidence)
    assert CryptographicCommitment.verify("P", salt, comm_hash) is False
    assert CryptographicCommitment.verify("S", salt, comm_hash) is False


def test_data_consent_and_erasure():
    """Verify GDPR-compliant opt-in consent and Right-to-Erasure."""
    cm = DataConsentManager()

    # Default is opted-out (ephemeral zero-storage)
    assert cm.consent_granted is False
    cm.log_round(1, "R", "P", "WIN", {"R": 0.5, "P": 0.3, "S": 0.2})
    assert len(cm.telemetry_log) == 0

    # User grants consent
    cm.set_consent(True)
    assert cm.consent_granted is True
    cm.log_round(1, "R", "P", "WIN", {"R": 0.5, "P": 0.3, "S": 0.2})
    cm.log_round(2, "P", "S", "WIN", {"R": 0.2, "P": 0.6, "S": 0.2})
    assert len(cm.telemetry_log) == 2

    # User exercises Right to be Forgotten (purge)
    purge_report = cm.purge_all_data()
    assert purge_report["status"] == "PURGED"
    assert purge_report["records_deleted"] == 2
    assert len(cm.telemetry_log) == 0


def test_explainability_card_generation():
    """Verify explainability documentation format."""
    probs = {"R": 0.70, "P": 0.15, "S": 0.15}
    eu = {"R": 0.0, "P": 0.55, "S": -0.55}
    reasoning = {
        "detected": True,
        "strategy": "WIN_STAY_LOSE_SHIFT",
        "explanation": "Opponent repeated Rock after winning.",
    }

    card = format_explainability_card(
        round_num=5,
        agent_move="P",
        probs=probs,
        eu=eu,
        reasoning=reasoning,
        entropy=1.18,
    )

    assert card["round"] == 5
    assert card["selected_action"] == "P"
    assert card["rule_detected"] is True
    assert card["active_strategy"] == "WIN_STAY_LOSE_SHIFT"
    assert "Rule Layer:" in card["explanation"]
