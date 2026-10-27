"""Phase 0 smoke test: proves the toolchain (pytest, ruff-clean imports) works."""

from pathlib import Path

import voc


def test_package_importable() -> None:
    assert voc.__doc__ is not None
    assert "Voice-of-Customer" in voc.__doc__


def test_repo_layout_matches_prd_section_15() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "pyproject.toml",
        "Makefile",
        "docker-compose.yml",
        ".env.example",
        "AGENTS.md",
        "docs/PRD.md",
        "flow.md",
        "decision.md",
        "src/voc/data",
        "src/voc/models",
        "src/voc/evaluation",
        "src/voc/api",
        "src/voc/rag",
        "src/voc/monitoring",
        "src/voc/dashboard",
        "sql/migrations",
        "sql/analytics",
        "configs",
        "eval",
        "results",
        "notebooks",
        "airflow/dags",
        "tests",
        ".github/workflows/ci.yml",
    ]
    missing = [rel for rel in required if not (root / rel).exists()]
    assert missing == [], f"Missing from repo layout: {missing}"
