from __future__ import annotations

import ast
import re
from typing import Iterable


def _normalize_lines(value) -> set[int]:
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return {int(v) for v in parsed}
        except Exception:
            pass

        return {int(match) for match in re.findall(r"\d+", value)}

    return {int(value)}


def localization_accuracy(predicted, actual):
    return int(_normalize_lines(predicted) == _normalize_lines(actual))


def hallucination_rate(output_text, ground_truth_fix):
    return int(ground_truth_fix.lower() not in str(output_text).lower())
