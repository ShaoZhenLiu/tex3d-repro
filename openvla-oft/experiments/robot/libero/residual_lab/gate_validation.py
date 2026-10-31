"""Audit selective filtering without penalizing absent groups or error ratios."""
import math


def candidate_passes(candidate, thresholds):
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    error = candidate.get('q_err')
    signal = candidate.get('q_signal')
    if not finite(error) or error > thresholds['max_q_err']:
        return False
    if not finite(signal) or signal < thresholds['min_q_signal']:
        return False
    for key, value in candidate.items():
        if key.startswith('by_') and key.endswith('.q_signal'):
            # None explicitly denotes a group with no observations.
            if value is not None and (
                    not finite(value) or value < thresholds['min_group_q_signal']):
                return False
    return True
