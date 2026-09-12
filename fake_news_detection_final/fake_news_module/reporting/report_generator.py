"""PDF report generation for fake news analyses."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

REPORT_TITLE = "Fake News Detector — AI Evidence Integrity Monitor"
FOOTER_TEXT = "Generated automatically by AI Evidence Integrity Monitor"


def build_report_payload(result: Mapping[str, Any], *, analysis_id: str | None = None) -> dict[str, Any]:
    """Normalize a pipeline result into a PDF-ready report payload."""
    result = dict(result or {})
    explanation = dict(result.get("explanation") or {})
    evidence = dict(explanation.get("evidence") or {})
    confidence_breakdown = dict(explanation.get("confidence_breakdown") or {})
    api_results = dict(result.get("api_results") or {})
    timestamp = result.get("analysis_timestamp") or result.get("timestamp") or datetime.now(timezone.utc).isoformat()
    score = _as_float(result.get("score"), default=0.0)
    confidence_pct = explanation.get("confidence_pct")
    if confidence_pct is None:
        confidence_pct = max(1, min(100, round(abs(score) * 100)))

    similar_claims = _extract_similar_claims(evidence)
    source_evidence = evidence.get("source_evidence") or {}

    payload = {
        "analysis_id": analysis_id or result.get("analysis_id") or "",
        "timestamp": timestamp,
        "claim": str(result.get("claim") or "N/A"),
        "final_verdict": str(result.get("final_decision") or explanation.get("verdict") or "Uncertain"),
        "confidence": str(result.get("confidence") or explanation.get("confidence") or "Low"),
        "confidence_pct": confidence_pct,
        "score": score,
        "roberta_prediction": str(api_results.get("roberta") or evidence.get("roberta_evidence", {}).get("label") or "Unknown"),
        "similar_claims": similar_claims,
        "fact_check": dict(evidence.get("factcheck_evidence") or {}),
        "source_credibility": {
            "source_score": source_evidence.get("source_score"),
            "sources_found": list(source_evidence.get("sources_found") or []),
            "description": source_evidence.get("description", ""),
        },
        "evidence_summary": _build_evidence_summary(evidence),
        "reasons": list(explanation.get("reasons") or []),
        "confidence_breakdown": confidence_breakdown,
        "raw_explanation": explanation,
        "api_results": api_results,
    }
    return payload


def generate_pdf_report(report_data: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render a professional PDF report to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=REPORT_TITLE,
        author="AI Evidence Integrity Monitor",
        subject="Fake news analysis report",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_draw_footer)])

    story = []
    story.append(Paragraph(REPORT_TITLE, styles["TitleAccent"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Analysis Report", styles["Subtitle"]))
    story.append(Spacer(1, 8))
    story.append(_kv_table([
        ("Timestamp", str(report_data.get("timestamp") or "N/A")),
        ("Claim", str(report_data.get("claim") or "N/A")),
        ("Final Verdict", str(report_data.get("final_verdict") or "Uncertain")),
        ("Confidence Score", f'{report_data.get("confidence_pct", 0)}%'),
        ("RoBERTa Prediction", str(report_data.get("roberta_prediction") or "Unknown")),
    ], styles))
    story.append(Spacer(1, 10))
    story.append(_section_header("Similar Claims", styles))
    story.append(_similar_claims_table(report_data.get("similar_claims") or [], styles))
    story.append(Spacer(1, 10))
    story.append(_section_header("Fact Check Results", styles))
    story.append(_factcheck_table(report_data.get("fact_check") or {}, styles))
    story.append(Spacer(1, 10))
    story.append(_section_header("Source Credibility", styles))
    story.append(_source_table(report_data.get("source_credibility") or {}, styles))
    story.append(Spacer(1, 10))
    story.append(_section_header("Evidence Summary", styles))
    story.append(_bullet_list(report_data.get("evidence_summary") or [], styles, empty_text="No structured evidence summary available."))
    story.append(Spacer(1, 10))
    story.append(_section_header("Explainability Reasons", styles))
    story.append(_bullet_list(report_data.get("reasons") or [], styles, empty_text="No explainability reasons available."))
    story.append(Spacer(1, 10))
    story.append(_section_header("Confidence Breakdown", styles))
    story.append(_confidence_table(report_data.get("confidence_breakdown") or {}, styles))

    doc.build(story)
    logger.info("PDF report generated at %s", output_path)
    return output_path


def cleanup_old_reports(report_dir: str | Path, *, max_age_hours: int = 24, keep_latest: int = 100) -> int:
    """Delete stale PDF reports from the temporary reports directory."""
    report_dir = Path(report_dir)
    if not report_dir.exists():
        return 0

    cutoff_seconds = max_age_hours * 3600
    pdf_files = sorted(
        [path for path in report_dir.glob("*.pdf") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    now = datetime.now().timestamp()
    removed = 0

    for index, path in enumerate(pdf_files):
        age_seconds = now - path.stat().st_mtime
        if age_seconds > cutoff_seconds or index >= keep_latest:
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError as exc:
                logger.debug("Could not remove stale report %s: %s", path, exc)

    return removed


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "TitleAccent": ParagraphStyle(
            "TitleAccent",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1d4ed8"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#4b5563"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "Section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#111827"),
            spaceBefore=4,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=12,
            textColor=colors.HexColor("#1f2937"),
        ),
        "BodySmall": ParagraphStyle(
            "BodySmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=11,
            textColor=colors.HexColor("#374151"),
        ),
        "ListItem": ParagraphStyle(
            "ListItem",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11.2,
            textColor=colors.HexColor("#1f2937"),
            leftIndent=0,
            firstLineIndent=0,
            alignment=TA_LEFT,
        ),
    }
    return styles


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawCentredString(doc.pagesize[0] / 2.0, 10 * mm, FOOTER_TEXT)
    canvas.restoreState()


def _section_header(title: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(title, styles["Section"])


def _kv_table(rows: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    table_data = [[Paragraph(f"<b>{label}</b>", styles["BodySmall"]), Paragraph(value, styles["Body"])] for label, value in rows]
    table = Table(table_data, colWidths=[38 * mm, 130 * mm], repeatRows=0)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dbe4f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _similar_claims_table(similar_claims: list[dict[str, Any]], styles: dict[str, ParagraphStyle]) -> Table:
    if not similar_claims:
        return _single_cell_table("No strongly similar past claims found.", styles)

    rows = [[
        Paragraph("<b>Title</b>", styles["BodySmall"]),
        Paragraph("<b>Similarity</b>", styles["BodySmall"]),
        Paragraph("<b>Label</b>", styles["BodySmall"]),
    ]]
    for item in similar_claims[:5]:
        rows.append([
            Paragraph(_safe_text(item.get("title") or item.get("claim") or "Untitled"), styles["Body"]),
            Paragraph(f"{_as_float(item.get('score'), default=0.0):.2f}", styles["Body"]),
            Paragraph(_safe_text(item.get("label") or "Unknown"), styles["Body"]),
        ])
    table = Table(rows, colWidths=[112 * mm, 25 * mm, 25 * mm], repeatRows=1)
    table.setStyle(_table_style(header_bg="#dbeafe"))
    return table


def _factcheck_table(fact_check: Mapping[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    rows = [
        [Paragraph("<b>Source</b>", styles["BodySmall"]), Paragraph(_safe_text(fact_check.get("source") or "Google Fact Check Tools API"), styles["Body"])],
        [Paragraph("<b>Label</b>", styles["BodySmall"]), Paragraph(_safe_text(fact_check.get("label") or "Unknown"), styles["Body"])],
        [Paragraph("<b>Description</b>", styles["BodySmall"]), Paragraph(_safe_text(fact_check.get("description") or "No fact-check evidence available."), styles["Body"])],
    ]
    table = Table(rows, colWidths=[40 * mm, 122 * mm], repeatRows=0)
    table.setStyle(_table_style())
    return table


def _source_table(source_credibility: Mapping[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    sources = source_credibility.get("sources_found") or []
    source_score = source_credibility.get("source_score")
    rows = [
        [Paragraph("<b>Average Score</b>", styles["BodySmall"]), Paragraph(f"{_as_float(source_score, default=0.5):.2f}", styles["Body"])],
        [Paragraph("<b>Description</b>", styles["BodySmall"]), Paragraph(_safe_text(source_credibility.get("description") or "No source credibility evidence available."), styles["Body"])],
    ]
    if sources:
        domain_lines = []
        for item in sources[:5]:
            if isinstance(item, Mapping):
                domain = _safe_text(item.get("domain") or "Unknown")
                score = _as_float(item.get("score"), default=0.5)
                domain_lines.append(f"{domain} ({score:.2f})")
            else:
                domain_lines.append(_safe_text(str(item)))
        rows.append([Paragraph("<b>Sources Found</b>", styles["BodySmall"]), Paragraph("<br/>".join(domain_lines), styles["Body"])])
    else:
        rows.append([Paragraph("<b>Sources Found</b>", styles["BodySmall"]), Paragraph("None", styles["Body"])])
    table = Table(rows, colWidths=[40 * mm, 122 * mm], repeatRows=0)
    table.setStyle(_table_style())
    return table


def _confidence_table(confidence_breakdown: Mapping[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    if not confidence_breakdown:
        return _single_cell_table("No confidence breakdown available.", styles)

    rows = [[
        Paragraph("<b>Signal</b>", styles["BodySmall"]),
        Paragraph("<b>Weight</b>", styles["BodySmall"]),
        Paragraph("<b>Label</b>", styles["BodySmall"]),
        Paragraph("<b>Contributed</b>", styles["BodySmall"]),
        Paragraph("<b>Direction</b>", styles["BodySmall"]),
    ]]
    for signal, data in confidence_breakdown.items():
        if not isinstance(data, Mapping):
            continue
        rows.append([
            Paragraph(_safe_text(signal), styles["Body"]),
            Paragraph(f"{int(data.get('weight_pct', 0))}%", styles["Body"]),
            Paragraph(_safe_text(data.get("label") or "Unknown"), styles["Body"]),
            Paragraph("Yes" if data.get("contributed") else "No", styles["Body"]),
            Paragraph(_safe_text(data.get("direction") or "NEUTRAL"), styles["Body"]),
        ])
    table = Table(rows, colWidths=[38 * mm, 22 * mm, 40 * mm, 24 * mm, 30 * mm], repeatRows=1)
    table.setStyle(_table_style(header_bg="#ecfeff"))
    return table


def _bullet_list(items: list[Any], styles: dict[str, ParagraphStyle], *, empty_text: str) -> ListFlowable | Table:
    if not items:
        return _single_cell_table(empty_text, styles)

    flow_items = []
    for item in items:
        if isinstance(item, Mapping):
            text = ", ".join(f"{key}: {value}" for key, value in item.items())
        else:
            text = str(item)
        flow_items.append(ListItem(Paragraph(_safe_text(text), styles["ListItem"])))
    return ListFlowable(flow_items, bulletType="bullet", leftIndent=12)


def _single_cell_table(text: str, styles: dict[str, ParagraphStyle]) -> Table:
    table = Table([[Paragraph(_safe_text(text), styles["Body"])]], colWidths=[162 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _table_style(*, header_bg: str = "#e0f2fe") -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dbe4f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ])


def _extract_similar_claims(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    matches = (evidence.get("similarity_evidence") or {}).get("top_matches") or []
    output = []
    for match in matches[:5]:
        if not isinstance(match, Mapping):
            continue
        output.append({
            "title": match.get("title") or match.get("claim") or "Untitled",
            "score": _as_float(match.get("score"), default=0.0),
            "label": match.get("label") or "Unknown",
        })
    return output


def _build_evidence_summary(evidence: Mapping[str, Any]) -> list[str]:
    summary: list[str] = []
    for key in ("roberta_evidence", "similarity_evidence", "factcheck_evidence", "newsapi_evidence", "groq_evidence", "source_evidence"):
        item = evidence.get(key) or {}
        if isinstance(item, Mapping):
            description = item.get("description")
            if description:
                summary.append(f"{key.replace('_', ' ').title()}: {description}")
    return summary


def _safe_text(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)
