
from pipeline.metadata_pipeline import run_metadata_pipeline

def analyze_evidence(file_path, evidence_id):
    result = run_metadata_pipeline(file_path)

    return {
        "evidence_id": evidence_id,
        "metadata_score": result["score"],
        "flags": result["flags"],
        "timeline_flags": result["timeline_flags"],
        "anomaly": result["anomaly"]
    }
