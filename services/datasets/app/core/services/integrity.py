from collections import defaultdict
from typing import Dict, List, Tuple

from app.api.schemas.integrity import (
    IntegritySummary, LeakageSummary,
    DuplicateGroup, LeakageGroup, IntegrityReportResponse
)

SPLIT_PAIRS: List[Tuple[str, str]] = [("train", "val"), ("train", "test"), ("val", "test")]

def _group_by_split(hashes: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    """Группирует карту хешей: сплит -> хеш -> список файлов"""
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for path, file_hash in hashes.items():
        split = path.split("/", 1)[0]
        grouped[split][file_hash].append(path)
    return grouped

def compute_integrity_summary(hashes: Dict[str, str]) -> IntegritySummary:
    """Сводка по дубликатам внутри сплитов и утечкам между сплитами"""
    grouped = _group_by_split(hashes)

    duplicates_count = sum(
        len(files) - 1
        for split_hashes in grouped.values()
        for files in split_hashes.values()
        if len(files) > 1
    )

    leakage = {
        f"{a}_{b}": len(set(grouped.get(a, {})) & set(grouped.get(b, {})))
        for a, b in SPLIT_PAIRS
    }

    return IntegritySummary(
        duplicates_count=duplicates_count,
        leakage=LeakageSummary(**leakage)
    )

def build_integrity_report(
    dataset_id: str,
    version_id: str,
    hashes: Dict[str, str]
) -> IntegrityReportResponse:
    """Детальный отчёт: группы дубликатов и конкретные файлы утечек"""
    grouped = _group_by_split(hashes)

    duplicates = [
        DuplicateGroup(hash=file_hash, split=split, files=sorted(files))
        for split, split_hashes in grouped.items()
        for file_hash, files in split_hashes.items()
        if len(files) > 1
    ]

    leakage: Dict[str, List[LeakageGroup]] = {}
    for a, b in SPLIT_PAIRS:
        common = set(grouped.get(a, {})) & set(grouped.get(b, {}))
        leakage[f"{a}_{b}"] = [
            LeakageGroup(
                hash=file_hash,
                files={
                    a: sorted(grouped[a][file_hash]),
                    b: sorted(grouped[b][file_hash])
                }
            )
            for file_hash in sorted(common)
        ]

    return IntegrityReportResponse(
        dataset_id=dataset_id,
        version_id=version_id,
        summary=compute_integrity_summary(hashes),
        duplicates=duplicates,
        leakage=leakage
    )
