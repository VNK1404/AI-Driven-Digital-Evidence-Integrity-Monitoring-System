"""Flask routes for the analytics dashboard."""

from __future__ import annotations

from flask import Blueprint, jsonify

from fake_news_module.analytics.analytics_service import get_analytics_service

analytics_bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")


@analytics_bp.route("", methods=["GET"])
def analytics_root():
    service = get_analytics_service()
    return jsonify({
        "summary": service.summary(),
        "performance": service.performance(),
    })


@analytics_bp.route("/summary", methods=["GET"])
def analytics_summary():
    return jsonify(get_analytics_service().summary())


@analytics_bp.route("/performance", methods=["GET"])
def analytics_performance():
    return jsonify(get_analytics_service().performance())
