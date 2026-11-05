"""Resolve the paper's suite labels to the installed LIBERO registry."""

SUITE_IDS = {'spatial': 'libero_spatial', 'object': 'libero_object',
             'goal': 'libero_goal', 'long': 'libero_10'}


def create_suites(registry, names):
    identifiers = {name: SUITE_IDS[name] for name in names}
    missing = set(identifiers.values()).difference(registry)
    if missing:
        raise KeyError('Missing LIBERO suites: ' + ', '.join(sorted(missing)))
    return {name: registry[identifier]() for name, identifier in identifiers.items()}
