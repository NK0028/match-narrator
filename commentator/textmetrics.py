"""Dependency-free BLEU, ROUGE-L, CIDEr-D and METEOR-lite for commentary lines."""
from __future__ import annotations

import math
import re
from collections import Counter


def tok(s: str) -> list[str]:
    s = re.sub(r"\[(player|team|referee|coach)[^\]]*\]", r" \1 ", s.lower())
    return re.findall(r"[a-z0-9']+", s)


def ngrams(t: list[str], n: int) -> Counter:
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def corpus_bleu(hyps: list[list[str]], refs: list[list[list[str]]], max_n: int = 4) -> dict:
    p_num = [0] * max_n
    p_den = [0] * max_n
    hl = rl = 0
    for h, rs in zip(hyps, refs):
        hl += len(h)
        rl += min((abs(len(r) - len(h)), len(r)) for r in rs)[1]
        for n in range(1, max_n + 1):
            hc = ngrams(h, n)
            mx: Counter = Counter()
            for r in rs:
                mx |= ngrams(r, n)
            p_num[n - 1] += sum(min(c, mx[g]) for g, c in hc.items())
            p_den[n - 1] += max(sum(hc.values()), 0)
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    out = {}
    for n in (1, 4):
        ps = [(p_num[i] + 1e-9) / (p_den[i] + 1e-9) for i in range(n)]
        out[f"BLEU-{n}"] = bp * math.exp(sum(math.log(p) for p in ps) / n)
    return out


def rouge_l(h: list[str], r: list[str], beta: float = 1.2) -> float:
    if not h or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i, a in enumerate(h):
        for j, b in enumerate(r):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a == b else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, rc = lcs / len(h), lcs / len(r)
    return (1 + beta ** 2) * p * rc / (rc + beta ** 2 * p)


def meteor_lite(h: list[str], r: list[str]) -> float:
    """Exact-match METEOR (no stemming/synonyms): F-mean with fragmentation penalty."""
    if not h or not r:
        return 0.0
    used, align = set(), []
    for i, w in enumerate(h):
        for j, v in enumerate(r):
            if j not in used and w == v:
                used.add(j)
                align.append((i, j))
                break
    m = len(align)
    if m == 0:
        return 0.0
    p, rc = m / len(h), m / len(r)
    fmean = 10 * p * rc / (rc + 9 * p)
    chunks = 1 + sum(1 for (a, b), (c, d) in zip(align, align[1:]) if not (c == a + 1 and d == b + 1))
    return fmean * (1 - 0.5 * (chunks / m) ** 3)


def cider_d(hyps: list[list[str]], refs: list[list[list[str]]], n: int = 4, sigma: float = 6.0) -> float:
    docs = [[ngrams(r, k) for k in range(1, n + 1)] for rs in refs for r in rs]
    df: list[Counter] = [Counter() for _ in range(n)]
    for rs in refs:
        seen = [set() for _ in range(n)]
        for r in rs:
            for k in range(n):
                seen[k] |= set(ngrams(r, k + 1))
        for k in range(n):
            df[k].update(seen[k])
    N = max(len(refs), 1)
    del docs

    def vec(t, k):
        c = ngrams(t, k + 1)
        return {g: tf * math.log(N / max(1.0, df[k][g])) for g, tf in c.items()}

    scores = []
    for h, rs in zip(hyps, refs):
        s = 0.0
        for k in range(n):
            vh = vec(h, k)
            nh = math.sqrt(sum(v * v for v in vh.values()))
            acc = 0.0
            for r in rs:
                vr = vec(r, k)
                nr = math.sqrt(sum(v * v for v in vr.values()))
                dot = sum(min(vh[g], vr.get(g, 0)) * vr.get(g, 0) for g in vh)
                if nh and nr:
                    acc += dot / (nh * nr) * math.exp(-((len(h) - len(r)) ** 2) / (2 * sigma ** 2))
            s += acc / len(rs)
        scores.append(10 * s / n)
    return sum(scores) / max(len(scores), 1)


def all_metrics(hyps: list[str], refs: list[list[str]]) -> dict:
    H = [tok(h) for h in hyps]
    R = [[tok(r) for r in rs] for rs in refs]
    out = corpus_bleu(H, R)
    out["ROUGE-L"] = sum(max(rouge_l(h, r) for r in rs) for h, rs in zip(H, R)) / max(len(H), 1)
    out["METEOR-lite"] = sum(max(meteor_lite(h, r) for r in rs) for h, rs in zip(H, R)) / max(len(H), 1)
    out["CIDEr-D"] = cider_d(H, R)
    out["n"] = len(H)
    return out
