"""
Responsible AI & Fair-Play Layer: Cryptographic Commitment, Explainability & Privacy Governance
===============================================================================================
Addresses ethical dimensions of human-AI gameplay:
1. Fair-Play Cryptographic Commitment:
   - Proves zero-peeking: AI commits to move via SHA-256(move + nonce) before receiving human input.
2. Data Consent & Privacy (GDPR/CCPA compliant):
   - Explicit opt-in consent for telemetry logging.
   - Ephemeral in-memory execution by default.
   - Instant "Right to be Forgotten" session purge.
3. Explainability & Model Transparency:
   - Real-time audit trails explaining why moves were chosen.
"""

import hashlib
import json
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple


class CryptographicCommitment:
    """Zero-peeking cryptographic fair-play commitment scheme using SHA-256."""

    def __init__(self):
        self.pending_salt: Optional[str] = None
        self.pending_move: Optional[str] = None
        self.pending_commitment: Optional[str] = None

    def create_commitment(self, agent_move: str) -> str:
        """Commit to an agent move before the player inputs their move.

        Returns:
            SHA-256 hex digest commitment string.
        """
        salt = secrets.token_hex(16)  # 128-bit cryptographic nonce
        payload = f"{agent_move}:{salt}".encode("utf-8")
        commitment = hashlib.sha256(payload).hexdigest()

        self.pending_move = agent_move
        self.pending_salt = salt
        self.pending_commitment = commitment

        return commitment

    def reveal(self) -> Tuple[str, str, str]:
        """Reveal the committed move, salt, and original commitment for client-side verification."""
        if self.pending_move is None or self.pending_salt is None:
            raise ValueError("No commitment currently pending.")

        move = self.pending_move
        salt = self.pending_salt
        commitment = self.pending_commitment

        # Reset pending state
        self.pending_move = None
        self.pending_salt = None
        self.pending_commitment = None

        return move, salt, commitment

    @staticmethod
    def verify(move: str, salt: str, commitment: str) -> bool:
        """Independently verify that a revealed move and salt reproduce the commitment hash."""
        payload = f"{move}:{salt}".encode("utf-8")
        expected_hash = hashlib.sha256(payload).hexdigest()
        return secrets.compare_digest(expected_hash, commitment)


class DataConsentManager:
    """Manages player data privacy, explicit consent, and right-to-erasure compliance."""

    def __init__(self):
        self.consent_granted: bool = False
        self.session_id: str = secrets.token_hex(8)
        self.telemetry_log: List[Dict[str, Any]] = []

    def set_consent(self, granted: bool) -> None:
        """Update player data consent status."""
        self.consent_granted = granted
        if not granted:
            # Purge any previously captured telemetry if consent is withdrawn
            self.telemetry_log.clear()

    def log_round(
        self,
        round_num: int,
        human_move: str,
        agent_move: str,
        outcome: str,
        predicted_probs: Dict[str, float],
        detected_strategy: Optional[str] = None,
    ) -> None:
        """Log round telemetry only if consent has been explicitly granted."""
        if not self.consent_granted:
            return  # Ephemeral operation: zero logging

        entry = {
            "round": round_num,
            "timestamp": time.time(),
            "human_move": human_move,
            "agent_move": agent_move,
            "outcome": outcome,
            "predicted_probs": predicted_probs,
            "detected_strategy": detected_strategy,
        }
        self.telemetry_log.append(entry)

    def purge_all_data(self) -> Dict[str, Any]:
        """Right-to-be-Forgotten: Purge all session records and telemetry immediately."""
        count = len(self.telemetry_log)
        self.telemetry_log.clear()
        self.session_id = secrets.token_hex(8)
        return {
            "status": "PURGED",
            "records_deleted": count,
            "message": "All session telemetry and gameplay records have been permanently erased.",
        }

    def get_privacy_audit(self) -> Dict[str, Any]:
        """Return transparency report regarding stored player telemetry."""
        return {
            "session_id": self.session_id,
            "consent_status": "OPTED_IN" if self.consent_granted else "OPTED_OUT_EPHEMERAL",
            "total_records_stored": len(self.telemetry_log),
            "storage_mode": "In-Memory Telemetry" if self.consent_granted else "Pure Ephemeral Zero-Storage",
            "pii_collected": "None (Move tokens {R, P, S} only)",
        }


def format_explainability_card(
    round_num: int,
    agent_move: str,
    probs: Dict[str, float],
    eu: Dict[str, float],
    reasoning: Dict[str, Any],
    entropy: float,
) -> Dict[str, Any]:
    """Compile structured explainability documentation for an individual round."""
    rule_detected = reasoning.get("detected", False)
    strategy = reasoning.get("strategy", "None")
    rule_explanation = reasoning.get("explanation", "")

    rationale = []
    if rule_detected:
        rationale.append(f"Rule Layer: {rule_explanation}")
    else:
        rationale.append(
            f"Bayesian Layer: Predicted opponent distribution is "
            f"R:{probs['R']:.1%}, P:{probs['P']:.1%}, S:{probs['S']:.1%} "
            f"(Shannon Entropy: {entropy:.2f} bits)."
        )
        rationale.append(
            f"Decision Rationale: Playing '{agent_move}' maximizes expected utility "
            f"(EU: {eu[agent_move]:+.2f} vs R:{eu['R']:+.2f}, P:{eu['P']:+.2f}, S:{eu['S']:+.2f})."
        )

    return {
        "round": round_num,
        "selected_action": agent_move,
        "predicted_probabilities": probs,
        "expected_utilities": eu,
        "active_strategy": strategy,
        "rule_detected": rule_detected,
        "explanation": " ".join(rationale),
        "entropy_bits": entropy,
    }
