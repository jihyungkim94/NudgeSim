"""LIAR claim pool: loading, de-identification, stratification, placebo matching.

Plan section 6. Two entry points with identical downstream behaviour:

* ``ClaimPool.from_liar(path)``  -- the real UCSB LIAR release (TSV, 14 cols).
* ``ClaimPool.synthetic(seed)``  -- a deterministic surrogate pool used when the
  corpus is unavailable (e.g. an offline runner or CI).

Every pool carries a ``provenance`` string that is written into the episode log
and into every results table, so a surrogate run can never be mistaken for a
LIAR-backed one.

Preprocessing safeguard (plan section 6): real speaker names are replaced with
generic attributions in all agent-visible text, so the simulation never
generates novel false associations about identifiable individuals.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from nudgesim.game.actions import Claim, Veracity

# LIAR's six-way label set, split into the false side (injected misinformation)
# and the true side (placebo controls). "half-true" is excluded from both: it is
# neither a clean falsehood to inject nor a clean truth to protect.
FALSE_LABELS = ("pants-fire", "false", "barely-true")
TRUE_LABELS = ("true", "mostly-true")
EXCLUDED_LABELS = ("half-true",)

_LIAR_COLUMNS = 14

# Generic attributions used to replace identifiable speakers. Chosen so the
# rewritten statement stays grammatical and keeps its rhetorical force without
# naming anyone.
_ATTRIBUTIONS = (
    "a national politician",
    "a state legislator",
    "a senior official",
    "a party spokesperson",
    "a congressional candidate",
    "a governor",
    "an advocacy group",
    "a cable news host",
)

_TITLE_RE = re.compile(
    r"\b(President|Senator|Sen\.|Governor|Gov\.|Representative|Rep\.|Congressman|"
    r"Congresswoman|Mayor|Secretary|Justice|Dr\.|Mr\.|Mrs\.|Ms\.)\s+"
    r"([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){0,2})"
)


def _stable_choice(key: str, options: Sequence[str]) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def deidentify(text: str, speaker_vocabulary: Iterable[str] = ()) -> str:
    """Replace identifiable speakers in agent-visible text with generic roles.

    Two passes: an explicit vocabulary (the LIAR ``speaker`` column, which is the
    reliable source of the names that actually recur in this corpus), then a
    title-plus-name regex for names that appear only inside statement text.
    """
    out = text
    names = sorted({n.strip() for n in speaker_vocabulary if n and n.strip()}, key=len, reverse=True)
    for name in names:
        pretty = name.replace("-", " ").strip()
        if len(pretty) < 4:
            continue
        pattern = re.compile(rf"\b{re.escape(pretty)}\b", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(_stable_choice(pretty, _ATTRIBUTIONS), out)

    def _sub_title(match: re.Match[str]) -> str:
        return _stable_choice(match.group(2), _ATTRIBUTIONS)

    out = _TITLE_RE.sub(_sub_title, out)
    return re.sub(r"\s{2,}", " ", out).strip()


@dataclass(frozen=True)
class PlaceboPair:
    """A false claim and its topic- and length-matched true counterpart."""

    false_claim: Claim
    true_claim: Claim

    @property
    def length_ratio(self) -> float:
        a, b = self.false_claim.n_tokens, self.true_claim.n_tokens
        return min(a, b) / max(a, b) if max(a, b) else 1.0


@dataclass
class ClaimPool:
    """Stratified claim pool with matched placebo pairs."""

    claims: list[Claim]
    provenance: str
    placebo_pairs: list[PlaceboPair] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ views

    @property
    def false_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.is_false]

    @property
    def true_claims(self) -> list[Claim]:
        return [c for c in self.claims if not c.is_false]

    def by_severity(self, severity: str) -> list[Claim]:
        return [c for c in self.claims if c.severity == severity]

    def stratified_sample(self, n: int, rng: random.Random) -> list[Claim]:
        """Sample false claims balanced across severity strata (plan section 5.9)."""
        strata = [self.by_severity(s) for s in FALSE_LABELS]
        strata = [s for s in strata if s]
        if not strata:
            raise ValueError("pool contains no false claims to stratify")
        out: list[Claim] = []
        i = 0
        while len(out) < n:
            bucket = strata[i % len(strata)]
            out.append(bucket[rng.randrange(len(bucket))])
            i += 1
        return out

    def placebo_for(self, false_claim: Claim) -> Claim | None:
        for pair in self.placebo_pairs:
            if pair.false_claim.claim_id == false_claim.claim_id:
                return pair.true_claim
        return None

    def summary(self) -> dict[str, object]:
        return {
            "provenance": self.provenance,
            "n_claims": len(self.claims),
            "n_false": len(self.false_claims),
            "n_true": len(self.true_claims),
            "n_placebo_pairs": len(self.placebo_pairs),
            "severity_counts": {
                s: len(self.by_severity(s)) for s in (*FALSE_LABELS, *TRUE_LABELS)
            },
            "mean_placebo_length_ratio": (
                sum(p.length_ratio for p in self.placebo_pairs) / len(self.placebo_pairs)
                if self.placebo_pairs
                else None
            ),
            **self.meta,
        }

    # ----------------------------------------------------------------- loaders

    @classmethod
    def from_liar(
        cls,
        path: str | Path,
        *,
        max_per_label: int | None = 200,
        seed: int = 0,
    ) -> "ClaimPool":
        """Load the real LIAR release.

        ``path`` is either a directory containing train/valid/test.tsv or a
        single .tsv file. Column order follows the UCSB release: id, label,
        statement, subjects, speaker, job, state, party, five credit-history
        counts, context.
        """
        path = Path(path)
        files = (
            sorted(path.glob("*.tsv")) if path.is_dir() else [path]
        )
        if not files:
            raise FileNotFoundError(f"no LIAR .tsv files under {path}")

        rows: list[dict[str, str]] = []
        speakers: set[str] = set()
        for file in files:
            with file.open(newline="", encoding="utf-8") as fh:
                for record in csv.reader(fh, delimiter="\t"):
                    if len(record) < _LIAR_COLUMNS:
                        continue
                    speakers.add(record[4])
                    rows.append(
                        {
                            "id": record[0],
                            "label": record[1].strip().lower(),
                            "statement": record[2],
                            "subjects": record[3],
                            "speaker": record[4],
                        }
                    )

        rng = random.Random(seed)
        buckets: dict[str, list[Claim]] = {}
        for row in rows:
            label = row["label"]
            if label in EXCLUDED_LABELS or label not in (*FALSE_LABELS, *TRUE_LABELS):
                continue
            text = deidentify(row["statement"], speakers)
            topic = (row["subjects"].split(",")[0] or "unspecified").strip().lower()
            claim = Claim(
                claim_id=f"liar-{row['id'].replace('.json', '')}",
                text=text,
                veracity=Veracity.FALSE if label in FALSE_LABELS else Veracity.TRUE,
                severity=label,
                topic=topic,
                n_tokens=len(text.split()),
                meta={"source": "LIAR", "deidentified": True},
            )
            buckets.setdefault(label, []).append(claim)

        claims: list[Claim] = []
        for label, bucket in buckets.items():
            rng.shuffle(bucket)
            claims.extend(bucket if max_per_label is None else bucket[:max_per_label])

        pool = cls(
            claims=claims,
            provenance="LIAR (Wang, 2017), UCSB release, de-identified",
            meta={"source_files": [f.name for f in files], "n_rows_read": len(rows)},
        )
        pool.placebo_pairs = _match_placebos(pool, rng)
        return pool

    @classmethod
    def synthetic(cls, seed: int = 0, n_per_label: int = 40) -> "ClaimPool":
        """Deterministic surrogate pool for offline runs.

        The claims are templated, non-defamatory, and carry no real speakers or
        events. They preserve the structural properties the simulation actually
        uses -- severity stratum, topic, and token length -- and nothing else.
        """
        rng = random.Random(seed)
        topics = ["health", "economy", "elections", "energy", "education", "security"]
        false_frames = {
            "pants-fire": [
                "Officials have quietly confirmed that {subject} was cancelled outright last month.",
                "Every {subject} record from the past decade was destroyed before the review began.",
                "A leaked memo shows {subject} figures were fabricated from start to finish.",
            ],
            "false": [
                "The new {subject} rules apply to every household in the country immediately.",
                "Spending on {subject} has tripled since the current policy took effect.",
                "No independent audit of {subject} has ever been carried out.",
            ],
            "barely-true": [
                "The {subject} programme costs more per person than any comparable scheme.",
                "Most of the {subject} budget last year went unspent.",
                "The {subject} figures in the report were revised downward twice.",
            ],
        }
        true_frames = {
            "true": [
                "The {subject} report was published on schedule and is publicly available.",
                "An independent body reviews {subject} spending once every fiscal year.",
                "The {subject} rules were amended after a formal consultation period.",
            ],
            "mostly-true": [
                "Funding for {subject} rose modestly over the last reporting period.",
                "The {subject} review covered the majority of participating regions.",
                "Most {subject} applications were processed within the stated window.",
            ],
        }
        subjects = {
            "health": ["the vaccination programme", "the hospital funding plan", "the drug pricing rule"],
            "economy": ["the payroll tax change", "the small business credit", "the inflation adjustment"],
            "elections": ["the voter roll update", "the ballot deadline", "the district boundary review"],
            "energy": ["the grid upgrade", "the fuel subsidy", "the emissions target"],
            "education": ["the tuition cap", "the school meals scheme", "the teacher pay review"],
            "security": ["the border staffing plan", "the cyber incident report", "the port inspection rule"],
        }

        claims: list[Claim] = []
        for label_group in (false_frames, true_frames):
            for label, frames in label_group.items():
                for i in range(n_per_label):
                    topic = topics[i % len(topics)]
                    subject = rng.choice(subjects[topic])
                    text = rng.choice(frames).format(subject=subject)
                    claims.append(
                        Claim(
                            claim_id=f"syn-{label}-{i:03d}",
                            text=text,
                            veracity=Veracity.FALSE if label in FALSE_LABELS else Veracity.TRUE,
                            severity=label,
                            topic=topic,
                            n_tokens=len(text.split()),
                            meta={"source": "synthetic-surrogate", "deidentified": True},
                        )
                    )

        pool = cls(
            claims=claims,
            provenance="SYNTHETIC-SURROGATE (no LIAR data present; not a LIAR result)",
            meta={"n_per_label": n_per_label, "seed": seed},
        )
        pool.placebo_pairs = _match_placebos(pool, rng)
        return pool


# How arresting a statement is, by LIAR severity stratum. Vosoughi, Roy & Aral
# (2018) attribute the false-versus-true diffusion asymmetry to novelty and the
# emotional response it provokes, not to veracity as such: nobody forwards a
# claim because it is false, they forward it because it is startling. The
# surrogate needs this channel or false and true claims diffuse identically and
# the calibration gate in plan section 5.7 cannot be satisfied by construction.
# It is a declared property of the *claim text*, visible to agents in the same
# way the text is -- never a leak of the veracity label.
ENGAGEMENT_BY_SEVERITY: dict[str, float] = {
    "pants-fire": 1.00,
    "false": 0.75,
    "barely-true": 0.50,
    "mostly-true": 0.25,
    "true": 0.00,
}


def claim_engagement(claim: Claim) -> float:
    """Novelty/arousal value of a claim, in [0, 1]."""
    return ENGAGEMENT_BY_SEVERITY.get(claim.severity, 0.5)


def _match_placebos(pool: ClaimPool, rng: random.Random) -> list[PlaceboPair]:
    """Pair each false claim with a true claim matched on topic, then length.

    Plan section 5.9: "placebo claims matched on topic and length". Matching is
    greedy on length within topic and falls back to global length matching when
    a topic has no true claims.
    """
    by_topic: dict[str, list[Claim]] = {}
    for claim in pool.true_claims:
        by_topic.setdefault(claim.topic, []).append(claim)
    all_true = list(pool.true_claims)
    if not all_true:
        return []

    pairs: list[PlaceboPair] = []
    for false_claim in pool.false_claims:
        candidates = by_topic.get(false_claim.topic) or all_true
        best = min(candidates, key=lambda c: (abs(c.n_tokens - false_claim.n_tokens), c.claim_id))
        pairs.append(PlaceboPair(false_claim=false_claim, true_claim=best))
    return pairs
