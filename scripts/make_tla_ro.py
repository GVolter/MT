import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import ahocorasick

TLA_L = "<TLA>"
TLA_R = "</TLA>"


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iter_ron_terms(cell: str) -> List[str]:
    out = []
    for t in (cell or "").split(","):
        t = t.strip()
        t = t.strip(";:.!?\"'()[]{}")
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) >= 2:
            out.append(t)
    return out


def load_papahagi_dict(tsv_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("TSV has no header/fieldnames.")
        if "ron" not in reader.fieldnames or "rup" not in reader.fieldnames:
            raise ValueError(f"Expected columns 'ron' and 'rup' in TSV. Found: {reader.fieldnames}")

        for row in reader:
            rup = (row.get("rup") or "").strip()
            ron_cell = (row.get("ron") or "").strip()
            if not rup or not ron_cell:
                continue

            for ron_term in iter_ron_terms(ron_cell):
                pairs.append((ron_term, rup))

    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def build_automaton(pairs: List[Tuple[str, str]]) -> ahocorasick.Automaton:
    """
    Build Aho–Corasick automaton over normalized ron_term.
    Store (ron_term_original, rup_lemma, ron_term_norm_len) as value.
    """
    A = ahocorasick.Automaton()
    for ron_term, rup in pairs:
        key = normalize_text(ron_term)
        if not key:
            continue
        A.add_word(key, (ron_term, rup, len(key)))
    A.make_automaton()
    return A


def word_boundary_ok(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start - 1].isalnum() or (start > 0 and text[start - 1] == "_"):
        return False
    if end < len(text) and (text[end].isalnum() or text[end] == "_"):
        return False
    return True


def find_matches(line: str, automaton: ahocorasick.Automaton) -> List[Tuple[int, int, str]]:
    low = line.lower()
    matches: List[Tuple[int, int, str]] = []
    for end_idx, (_ron_orig, rup, key_len) in automaton.iter(low):
        start_idx = end_idx - key_len + 1
        end_excl = end_idx + 1

        if not word_boundary_ok(low, start_idx, end_excl):
            continue

        matches.append((start_idx, end_excl, rup))

    return matches


def select_non_overlapping(matches: List[Tuple[int, int, str]], max_ann: int) -> List[Tuple[int, int, str]]:
    if not matches or max_ann <= 0:
        return []

    # Sort by length desc, then start asc
    matches = sorted(matches, key=lambda x: (-(x[1] - x[0]), x[0]))

    chosen: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []

    def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    for s, e, rup in matches:
        interval = (s, e)
        if any(overlaps(interval, occ) for occ in occupied):
            continue
        chosen.append((s, e, rup))
        occupied.append(interval)
        if len(chosen) >= max_ann:
            break

    chosen.sort(key=lambda x: x[0])
    return chosen


def annotate_line(line: str, automaton: ahocorasick.Automaton, p_annot: float, max_ann: int, rng: random.Random) -> str:
    if p_annot <= 0.0 or rng.random() > p_annot:
        return line

    matches = find_matches(line, automaton)
    chosen = select_non_overlapping(matches, max_ann=max_ann)
    if not chosen:
        return line

    out = line
    for s, e, rup in reversed(chosen):
        insert = f"{out[s:e]} {TLA_L} {rup} {TLA_R}"
        out = out[:s] + insert + out[e:]
    return out


def prefilter_pairs_by_corpus(pairs: List[Tuple[str, str]], in_ro: Path, max_chars: int) -> List[Tuple[str, str]]:
    text = in_ro.read_text(encoding="utf-8", errors="ignore").lower()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]

    kept = []
    for ron_term, rup in pairs:
        key = ron_term.lower()
        if key and key in text:
            kept.append((ron_term, rup))
    kept.sort(key=lambda x: len(x[0]), reverse=True)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Create TLA-annotated Romanian source file using Papahagi dictionary.")
    ap.add_argument("--dict-tsv", type=Path, required=True)
    ap.add_argument("--in-ro", type=Path, required=True)
    ap.add_argument("--out-ro", type=Path, required=True)
    ap.add_argument("--p-annot", type=float, default=0.5)
    ap.add_argument("--max-ann", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument(
        "--prefilter",
        action="store_true",
        help="Prefilter dictionary to only terms appearing in the corpus (faster).",
    )
    ap.add_argument(
        "--prefilter-max-chars",
        type=int,
        default=5_000_000,
        help="Max characters to scan for prefilter (0 = scan full file).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print progress every N lines.",
    )

    args = ap.parse_args()
    rng = random.Random(args.seed)

    pairs = load_papahagi_dict(args.dict_tsv)
    print(f"Loaded dictionary pairs: {len(pairs)}")

    if args.prefilter:
        pairs = prefilter_pairs_by_corpus(pairs, args.in_ro, args.prefilter_max_chars)
        print(f"After prefilter: {len(pairs)} pairs kept")

    automaton = build_automaton(pairs)

    args.out_ro.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    annotated = 0
    with args.in_ro.open("r", encoding="utf-8") as fin, args.out_ro.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            new_line = annotate_line(line, automaton, args.p_annot, args.max_ann, rng)
            if new_line != line:
                annotated += 1
            fout.write(new_line + "\n")
            n += 1
            if args.progress_every > 0 and n % args.progress_every == 0:
                print(f"Processed {n} lines (annotated {annotated})")

    print(f"Done. Wrote: {args.out_ro}")
    print(f"Total lines: {n}, annotated: {annotated} ({annotated / max(1,n):.1%})")


if __name__ == "__main__":
    main()
