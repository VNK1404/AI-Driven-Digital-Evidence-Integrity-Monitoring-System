"""
blockchain.py
-------------
Blockchain ledger for the Evidence Integrity System.

KEY DESIGN:
  - Only MANUALLY APPROVED, verified files are added to the blockchain.
  - A file must first be uploaded and verified as UNTAMPERED.
  - An officer/admin must explicitly approve it via the /approve API.
  - Once on the blockchain, the record is permanent and tamper-evident.

This ensures the blockchain only contains TRUSTED, CLEAN evidence.
"""

import hashlib
import json
from datetime import datetime


class Block:
    """
    A single block in the evidence blockchain.

    Each block holds:
      - index          : Position in chain (0 = genesis)
      - timestamp      : When this block was created (UTC)
      - evidence_data  : The approved evidence record
      - approved_by    : Who approved this evidence
      - previous_hash  : Hash of the preceding block
      - hash           : This block's own SHA-256 hash
    """

    def __init__(self, index: int, evidence_data: dict, approved_by: str, previous_hash: str):
        self.index         = index
        self.timestamp     = datetime.utcnow().isoformat()
        self.evidence_data = evidence_data   # {evidence_id, file_name, file_type, hash_value}
        self.approved_by   = approved_by
        self.previous_hash = previous_hash
        self.hash          = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of this block's full contents."""
        block_content = json.dumps({
            "index":         self.index,
            "timestamp":     self.timestamp,
            "evidence_data": self.evidence_data,
            "approved_by":   self.approved_by,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(block_content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Serialize block to dictionary."""
        return {
            "index":         self.index,
            "timestamp":     self.timestamp,
            "evidence_data": self.evidence_data,
            "approved_by":   self.approved_by,
            "previous_hash": self.previous_hash,
            "hash":          self.hash,
        }


class Blockchain:
    """
    The Evidence Blockchain Ledger.

    Only verified + manually approved files are added here.
    Maintains a tamper-evident chain where any modification
    to a past block breaks the entire chain.
    """

    def __init__(self):
        self.chain: list[Block] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the first block — the anchor of the chain."""
        genesis = Block(
            index         = 0,
            evidence_data = {"info": "Genesis Block — Verified Evidence Ledger"},
            approved_by   = "system",
            previous_hash = "0",
        )
        self.chain.append(genesis)

    def get_latest_block(self) -> Block:
        """Return the most recently added block."""
        return self.chain[-1]

    def add_approved_block(self, evidence_data: dict, approved_by: str) -> Block:
        """
        Add a new block for a manually approved, verified evidence file.

        Args:
            evidence_data (dict): {
                evidence_id, file_name, file_type, hash_value, file_size
            }
            approved_by (str): Name/ID of the approving officer.

        Returns:
            Block: The newly created block.
        """
        new_block = Block(
            index         = len(self.chain),
            evidence_data = evidence_data,
            approved_by   = approved_by,
            previous_hash = self.get_latest_block().hash,
        )
        self.chain.append(new_block)
        return new_block

    def get_chain(self) -> list[dict]:
        """Return the full blockchain as a list of dicts."""
        return [block.to_dict() for block in self.chain]

    def is_chain_valid(self) -> bool:
        """
        Validate the entire chain for tampering.

        Checks:
          1. Each block's stored hash matches its recomputed hash.
          2. Each block's previous_hash matches the prior block's hash.

        Returns:
            bool: True if chain is intact, False if tampered.
        """
        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]

            if current.hash != current._compute_hash():
                return False

            if current.previous_hash != previous.hash:
                return False

        return True

    def find_block(self, evidence_id: str) -> dict | None:
        """
        Find a block by evidence_id.

        Args:
            evidence_id (str): Evidence ID to search for.

        Returns:
            dict | None: Block as dict if found, else None.
        """
        for block in self.chain[1:]:   # skip genesis
            if block.evidence_data.get("evidence_id") == evidence_id:
                return block.to_dict()
        return None


# Module-level singleton — shared across the entire Flask app lifecycle
evidence_blockchain = Blockchain()
