from mugalois.core.types import Triple


def _normalize(value: str) -> str:
    """Extract local name and normalize for comparison."""
    if "/" in value:
        value = value.rstrip("/").split("/")[-1]
    if "#" in value:
        value = value.split("#")[-1]
    if ":" in value:
        value = value.split(":")[-1]
    return value.replace("_", " ").lower().strip()


def filter_by_subject(T1: set[Triple], T2: set[Triple]) -> set[Triple]:
    """Garde les triples de T1 dont le sujet apparaît comme sujet dans T2."""
    subjects_T2 = {_normalize(t.s) for t in T2}
    return {t for t in T1 if _normalize(t.s) in subjects_T2}


def filter_by_object(T1: set[Triple], T2: set[Triple]) -> set[Triple]:
    """Garde les triples de T1 dont l'objet apparaît comme objet dans T2."""
    objects_T2 = {_normalize(t.o) for t in T2}
    return {t for t in T1 if _normalize(t.o) in objects_T2}


def filter_by_subject_object(T1: set[Triple], T2: set[Triple]) -> set[Triple]:
    """Garde les triples de T1 dont le sujet apparaît comme objet dans T2."""
    objects_T2 = {_normalize(t.o) for t in T2}
    return {t for t in T1 if _normalize(t.s) in objects_T2}