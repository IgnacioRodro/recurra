"""Separating two groups from a metrics table, evaluated honestly."""
from .evaluate import MODELS, SCORES, ClassificationReport, classify_groups, classify_timecourse
from .features import FAMILIES, build_features, select_features

__all__ = ["classify_groups", "classify_timecourse", "ClassificationReport", "build_features",
           "select_features", "FAMILIES", "MODELS", "SCORES"]
