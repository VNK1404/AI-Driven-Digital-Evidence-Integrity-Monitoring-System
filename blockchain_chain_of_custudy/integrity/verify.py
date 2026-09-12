"""
verify.py
---------
Integrity verification for all supported file types.

Workflow:
  1. File is uploaded → hash stored in DB
  2. Later, file is verified → current hash recomputed
  3. If hashes MATCH   → VERIFIED (file is clean)
  4. If hashes DIFFER  → TAMPERED (alert saved to DB)
  5. If VERIFIED + approved by officer → goes to blockchain

Tamper alerts are saved to the database for audit purposes.
"""

from integrity.hashing import generate_hash

# Status constants
STATUS_VERIFIED = "VERIFIED"
STATUS_TAMPERED = "TAMPERED"
STATUS_ERROR    = "ERROR"


def verify_integrity(file_path: str, stored_hash: str) -> dict:
    """
    Verify a file's integrity by comparing its current hash
    against the hash stored at upload time.

    Args:
        file_path   (str): Path to the file to verify.
        stored_hash (str): Original SHA-256 hash from the database.

    Returns:
        dict: {
            "status":       "VERIFIED" | "TAMPERED" | "ERROR",
            "current_hash": str | None,
            "stored_hash":  str,
            "message":      str,
            "tampered":     bool
        }
    """
    try:
        current_hash = generate_hash(file_path)
    except FileNotFoundError as e:
        return {
            "status":       STATUS_ERROR,
            "current_hash": None,
            "stored_hash":  stored_hash,
            "message":      f"File not found: {str(e)}",
            "tampered":     False,
        }
    except ValueError as e:
        return {
            "status":       STATUS_ERROR,
            "current_hash": None,
            "stored_hash":  stored_hash,
            "message":      f"Unsupported file type: {str(e)}",
            "tampered":     False,
        }
    except Exception as e:
        return {
            "status":       STATUS_ERROR,
            "current_hash": None,
            "stored_hash":  stored_hash,
            "message":      f"Unexpected error: {str(e)}",
            "tampered":     False,
        }

    # Compare hashes
    if current_hash == stored_hash:
        return {
            "status":       STATUS_VERIFIED,
            "current_hash": current_hash,
            "stored_hash":  stored_hash,
            "message":      "File integrity confirmed. This file is clean and unmodified.",
            "tampered":     False,
        }
    else:
        return {
            "status":       STATUS_TAMPERED,
            "current_hash": current_hash,
            "stored_hash":  stored_hash,
            "message": (
                "TAMPERING DETECTED! This file has been modified "
                "since it was originally registered. It cannot be trusted."
            ),
            "tampered": True,
        }
