"""
Credit scorecard development — Quantum project #3.

Builds a classic application scorecard the way a credit-risk team actually
builds one: coarse-classing every characteristic into bins, turning each bin
into a Weight of Evidence, ranking characteristics by Information Value,
fitting a logistic regression on the WOE-transformed inputs, and scaling the
log-odds into points (PDO / base-odds calibration).

Then it validates the thing honestly: out-of-sample Gini and KS, repeated
cross-validation confidence intervals (n=1000 is small — a single holdout
number would be noise), calibration by score band, PSI stability, a
cost-optimal cutoff under the dataset's official 5:1 cost matrix, and a
fairness check on the protected attributes the model deliberately excludes.

Data: Statlog German Credit (UCI), 1000 applications, 30% bad.
Only real dependencies are numpy/pandas/statsmodels (sklearn + matplotlib are
used for the challenger model and the chart, and are optional at import time).
"""

from __future__ import annotations

import json
import math
import os
import hashlib
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DATA_PATH = os.path.join(DATA_DIR, "german.data")
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
# The file is fetched over the network on first run, so the results on this
# page are only reproducible if it is the file they were built from. The
# digest is checked, not trusted: a mirror that quietly serves a different
# revision would otherwise change every number here without a word.
DATA_SHA256 = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age_years", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target",
]

NUMERIC = {
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "age_years", "existing_credits", "num_dependents",
}

# Characteristics a lender may not legally score on (sex, nationality).
# They stay in the data so we can audit the model's impact on those groups,
# but they never enter the scorecard. See README for the reasoning.
PROTECTED = {"personal_status_sex", "foreign_worker"}

PRETTY = {
    "checking_status": "Status of checking account",
    "duration_months": "Duration of credit (months)",
    "credit_history": "Credit history",
    "purpose": "Purpose of the loan",
    "credit_amount": "Credit amount (DM)",
    "savings": "Savings account / bonds",
    "employment_since": "Time in present employment",
    "installment_rate": "Instalment as % of disposable income",
    "personal_status_sex": "Personal status and sex",
    "other_debtors": "Other debtors / guarantors",
    "residence_since": "Years at present residence",
    "property": "Property owned",
    "age_years": "Age (years)",
    "other_installment_plans": "Other instalment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits at this bank",
    "job": "Job",
    "num_dependents": "People liable for maintenance",
    "telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

# UCI codes -> readable labels, so the scorecard is human-readable.
LEVELS = {
    "checking_status": {
        "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits / all paid", "A31": "all paid at this bank",
        "A32": "existing credits paid duly", "A33": "past delay",
        "A34": "critical account / credits elsewhere",
    },
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture",
        "A43": "radio/TV", "A44": "appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "other",
    },
    "savings": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">= 1000 DM", "A65": "unknown / none",
    },
    "employment_since": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
        "A74": "4-7 years", "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced", "A92": "female: div/sep/married",
        "A93": "male: single", "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors": {
        "A101": "none", "A102": "co-applicant", "A103": "guarantor",
    },
    "property": {
        "A121": "real estate", "A122": "savings agreement / life insurance",
        "A123": "car or other", "A124": "unknown / none",
    },
    "other_installment_plans": {
        "A141": "bank", "A142": "stores", "A143": "none",
    },
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident", "A173": "skilled employee",
        "A174": "management / self-employed",
    },
    "telephone": {"A191": "none", "A192": "registered"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

# Official UCI cost matrix: calling a bad applicant good costs 5x what
# turning away a good applicant costs.
COST_FALSE_GOOD = 5.0
COST_FALSE_BAD = 1.0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load German Credit, downloading and caching it on first run."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
            payload = resp.read()
        got = hashlib.sha256(payload).hexdigest()
        if got != DATA_SHA256:
            raise RuntimeError(
                f"{DATA_URL} returned a file this build does not recognise "
                f"(sha256 {got}, expected {DATA_SHA256}). Nothing was written. "
                "Check the source before trusting any number it produces.")
        with open(path, "wb") as fh:
            fh.write(payload)

    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    # UCI codes 1 = good, 2 = bad. Model the bad, as a risk team would.
    df["target"] = (df["target"] == 2).astype(int)
    for col, mapping in LEVELS.items():
        df[col] = df[col].map(lambda v: mapping.get(v, v))
    return df


# --------------------------------------------------------------------------
# Coarse classing: bins -> WOE -> IV
# --------------------------------------------------------------------------

@dataclass
class Bin:
    label: str
    count: int
    bads: int
    lo: float = -np.inf       # numeric bins
    hi: float = np.inf
    levels: tuple = ()        # categorical bins
    woe: float = 0.0
    iv: float = 0.0

    @property
    def goods(self) -> int:
        return self.count - self.bads

    @property
    def bad_rate(self) -> float:
        return self.bads / self.count if self.count else 0.0


@dataclass
class Binning:
    name: str
    kind: str                 # "numeric" | "categorical"
    bins: list = field(default_factory=list)
    iv: float = 0.0

    def transform(self, s: pd.Series) -> np.ndarray:
        """Map raw values to their bin's WOE."""
        out = np.zeros(len(s), dtype=float)
        if self.kind == "numeric":
            # Bins tile the real line, so the only values left at WOE 0 are
            # NaNs. A production scorecard gives missing values their own bin
            # with its own WOE; this dataset has none, so neutral is honest.
            vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            for b in self.bins:
                mask = (vals > b.lo) & (vals <= b.hi)
                out[mask] = b.woe
        else:
            vals = s.astype(object).to_numpy()
            lookup = {lvl: b.woe for b in self.bins for lvl in b.levels}
            # Unseen level -> WOE 0, i.e. neutral, the conservative default.
            out = np.array([lookup.get(v, 0.0) for v in vals], dtype=float)
        return out

    def bin_of(self, value):
        """Place one value in its bin, for scoring a single application.

        Unlike transform(), this refuses rather than guessing: scoring a batch
        can afford a neutral default for an odd row, but handing a credit
        officer a decision built on a value nobody could place is worse than
        telling them the application is incomplete.
        """
        if self.kind == "numeric":
            v = float(value)
            if not np.isfinite(v):
                raise ValueError(
                    f"{self.name}: missing or non-finite value cannot be scored")
            for b in self.bins:
                if b.lo < v <= b.hi:
                    return b
            raise ValueError(f"{self.name}: {v} falls outside every bin")
        for b in self.bins:
            if value in b.levels:
                return b
        raise ValueError(f"{self.name}: unseen level {value!r}")


def _woe_iv(bins: list, total_goods: int, total_bads: int) -> float:
    """Fill in WOE/IV per bin; return the characteristic's total IV.

    WOE = ln(%good / %bad), so a HIGHER WOE always means a SAFER bin.
    Empty cells get a 0.5 correction rather than an infinite WOE.
    """
    iv_total = 0.0
    for b in bins:
        g = b.goods if b.goods > 0 else 0.5
        d = b.bads if b.bads > 0 else 0.5
        pct_g = g / total_goods
        pct_b = d / total_bads
        b.woe = math.log(pct_g / pct_b)
        b.iv = (pct_g - pct_b) * b.woe
        iv_total += b.iv
    return iv_total


def bin_numeric(x: pd.Series, y: pd.Series, min_frac=0.05, max_bins=6,
                prebins=20) -> Binning:
    """Supervised binning: quantile pre-bins, merged until every bin is big
    enough, the bad rate is monotone in x, and there are at most max_bins.

    Monotonicity is the scorecard convention: it stops the model from telling
    a credit officer that risk goes up, then down, then up again with income.
    """
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yy = y.to_numpy(dtype=int)
    n = len(vals)
    min_count = max(int(round(min_frac * n)), 20)

    qs = np.unique(np.nanquantile(vals, np.linspace(0, 1, prebins + 1)))
    edges = list(qs[1:-1]) if len(qs) > 2 else []
    cuts = [-np.inf] + edges + [np.inf]

    def build(cuts):
        out = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            m = (vals > lo) & (vals <= hi)
            out.append(Bin(label="", count=int(m.sum()),
                           bads=int(yy[m].sum()), lo=lo, hi=hi))
        return out

    bins = build(cuts)
    bins = [b for b in bins if b.count > 0]
    # Dropping an empty pre-bin must not leave a hole between its neighbours:
    # an applicant landing in the hole would be scored one way by transform()
    # and another by bin_of(). Hand the vacated range to the next bin so the
    # bins always tile the real line.
    for prev, nxt in zip(bins[:-1], bins[1:]):
        nxt.lo = prev.hi
    if bins:
        bins[0].lo, bins[-1].hi = -np.inf, np.inf

    def merge(i):
        """Merge bin i with bin i+1."""
        a, b = bins[i], bins[i + 1]
        bins[i] = Bin(label="", count=a.count + b.count, bads=a.bads + b.bads,
                      lo=a.lo, hi=b.hi)
        del bins[i + 1]

    # 1. every bin needs enough volume to estimate a bad rate at all
    while len(bins) > 2:
        small = [i for i, b in enumerate(bins) if b.count < min_count]
        if not small:
            break
        i = min(small, key=lambda i: bins[i].count)
        if i == 0:
            merge(0)
        elif i == len(bins) - 1:
            merge(len(bins) - 2)
        else:  # merge into whichever neighbour has the closer bad rate
            left = abs(bins[i].bad_rate - bins[i - 1].bad_rate)
            right = abs(bins[i].bad_rate - bins[i + 1].bad_rate)
            merge(i - 1 if left <= right else i)

    # 2. force the bad rate to move in one direction
    with np.errstate(invalid="ignore"):
        mids = np.array([b.bad_rate for b in bins])
    direction = 1.0 if (len(bins) > 1 and mids[-1] >= mids[0]) else -1.0
    while len(bins) > 2:
        rates = [b.bad_rate for b in bins]
        viol = [i for i in range(len(rates) - 1)
                if direction * (rates[i + 1] - rates[i]) < 0]
        if not viol:
            break
        # merge the violating pair that costs the least information
        i = min(viol, key=lambda i: abs(rates[i + 1] - rates[i]))
        merge(i)

    # 3. cap the number of bins
    while len(bins) > max_bins:
        rates = [b.bad_rate for b in bins]
        i = min(range(len(rates) - 1), key=lambda i: abs(rates[i + 1] - rates[i]))
        merge(i)

    for b in bins:
        lo_txt = "-inf" if b.lo == -np.inf else _fmt_num(b.lo)
        hi_txt = "inf" if b.hi == np.inf else _fmt_num(b.hi)
        if b.lo == -np.inf:
            b.label = f"<= {hi_txt}"
        elif b.hi == np.inf:
            b.label = f"> {lo_txt}"
        else:
            b.label = f"({lo_txt}, {hi_txt}]"

    binning = Binning(name=str(x.name), kind="numeric", bins=bins)
    binning.iv = _woe_iv(bins, int((yy == 0).sum()), int(yy.sum()))
    return binning


def _fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def bin_categorical(x: pd.Series, y: pd.Series, min_frac=0.05,
                    max_bins=6) -> Binning:
    """Coarse-classing for nominal characteristics: order the levels by bad
    rate, then merge neighbours in that order until each group carries enough
    volume and there are at most max_bins groups."""
    xx = x.astype(object).to_numpy()
    yy = y.to_numpy(dtype=int)
    n = len(xx)
    min_count = max(int(round(min_frac * n)), 20)

    groups = []
    for lvl in pd.unique(xx):
        m = xx == lvl
        groups.append(Bin(label=str(lvl), count=int(m.sum()),
                          bads=int(yy[m].sum()), levels=(lvl,)))
    groups.sort(key=lambda b: b.bad_rate)

    def merge(i):
        a, b = groups[i], groups[i + 1]
        groups[i] = Bin(label="", count=a.count + b.count,
                        bads=a.bads + b.bads, levels=a.levels + b.levels)
        del groups[i + 1]

    while len(groups) > 2:
        small = [i for i, b in enumerate(groups) if b.count < min_count]
        if not small:
            break
        i = min(small, key=lambda i: groups[i].count)
        if i == 0:
            merge(0)
        elif i == len(groups) - 1:
            merge(len(groups) - 2)
        else:
            left = abs(groups[i].bad_rate - groups[i - 1].bad_rate)
            right = abs(groups[i].bad_rate - groups[i + 1].bad_rate)
            merge(i - 1 if left <= right else i)

    while len(groups) > max_bins:
        rates = [b.bad_rate for b in groups]
        i = min(range(len(rates) - 1), key=lambda i: abs(rates[i + 1] - rates[i]))
        merge(i)

    for b in groups:
        b.label = " / ".join(str(l) for l in b.levels)

    binning = Binning(name=str(x.name), kind="categorical", bins=groups)
    binning.iv = _woe_iv(groups, int((yy == 0).sum()), int(yy.sum()))
    return binning


def bin_all(df: pd.DataFrame, y: pd.Series, features: list) -> dict:
    out = {}
    for col in features:
        if col in NUMERIC:
            out[col] = bin_numeric(df[col], y)
        else:
            out[col] = bin_categorical(df[col], y)
    return out


def woe_frame(df: pd.DataFrame, binnings: dict, cols: list) -> pd.DataFrame:
    return pd.DataFrame(
        {c: binnings[c].transform(df[c]) for c in cols}, index=df.index
    )


def iv_strength(iv: float) -> str:
    """Siddiqi's rule of thumb for reading an Information Value."""
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspiciously strong"


# --------------------------------------------------------------------------
# Model: logistic regression on WOE, then stepwise selection
# --------------------------------------------------------------------------

def _fit_logit(X: pd.DataFrame, y: pd.Series):
    import statsmodels.api as sm
    return sm.Logit(y.to_numpy(dtype=float),
                    sm.add_constant(X.to_numpy(dtype=float), has_constant="add"),
                    ).fit(disp=0, method="newton", maxiter=100)


def select_features(Xw: pd.DataFrame, y: pd.Series, candidates: list,
                    p_enter=0.05, p_stay=0.10) -> tuple:
    """Forward stepwise with a sign constraint.

    The sign constraint is what separates a scorecard from a black box: with
    WOE = ln(%good/%bad) and a model of the bad, every coefficient must be
    negative. A positive one means the multivariate fit has flipped the
    characteristic's meaning, which no credit officer will sign off on.
    """
    selected, log = [], []
    remaining = list(candidates)

    while remaining:
        best = None
        for col in remaining:
            cols = selected + [col]
            try:
                res = _fit_logit(Xw[cols], y)
            except Exception:
                continue
            coef, pval = res.params[-1], res.pvalues[-1]
            if coef >= 0:                      # wrong sign -> not admissible
                continue
            if pval > p_enter:
                continue
            if best is None or pval < best[1]:
                best = (col, pval, res)
        if best is None:
            break
        col, pval, res = best
        selected.append(col)
        log.append({"step": len(selected), "entered": col,
                    "p_value": float(pval), "pseudo_r2": float(res.prsquared)})
        remaining.remove(col)

        # backward pass: a variable can become redundant once others enter
        while len(selected) > 1:
            res = _fit_logit(Xw[selected], y)
            pvals = res.pvalues[1:]
            coefs = res.params[1:]
            worst = int(np.argmax(pvals))
            if pvals[worst] > p_stay or coefs[worst] >= 0:
                dropped = selected.pop(worst)
                log.append({"step": len(selected), "removed": dropped,
                            "p_value": float(pvals[worst])})
                remaining.append(dropped)
            else:
                break

    return selected, log


# --------------------------------------------------------------------------
# Scaling log-odds into points
# --------------------------------------------------------------------------

@dataclass
class Scaling:
    pdo: float = 20.0          # points to double the odds
    base_score: float = 600.0
    base_odds: float = 50.0    # good:bad at base_score

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)


def build_scorecard(binnings: dict, cols: list, params: np.ndarray,
                    scaling: Scaling) -> dict:
    """Turn the fitted coefficients into a points table.

    ln(odds_good) = -(a + sum b_j * WOE_j)
    score         = offset + factor * ln(odds_good)
    so each bin contributes  -(b_j * WOE_j + a/n) * factor + offset/n
    and the points add up to the score.
    """
    intercept, coefs = float(params[0]), np.asarray(params[1:], dtype=float)
    n = len(cols)
    f, off = scaling.factor, scaling.offset
    card = {}
    for col, b_j in zip(cols, coefs):
        rows = []
        for b in binnings[col].bins:
            pts = -(b_j * b.woe + intercept / n) * f + off / n
            rows.append({
                "bin": b.label, "count": b.count, "bads": b.bads,
                "bad_rate": round(b.bad_rate, 4), "woe": round(b.woe, 4),
                "iv": round(b.iv, 4), "points": int(round(pts)),
            })
        card[col] = {"pretty": PRETTY.get(col, col),
                     "coefficient": round(float(b_j), 4),
                     "iv": round(binnings[col].iv, 4), "rows": rows}
    return card


def score_from_card(card: dict, applicant: dict, binnings: dict) -> dict:
    """Score one applicant off the points table (the way a branch would)."""
    breakdown, total = [], 0
    for col, entry in card.items():
        b = binnings[col].bin_of(applicant[col])
        row = next(r for r in entry["rows"] if r["bin"] == b.label)
        total += row["points"]
        breakdown.append({"characteristic": entry["pretty"], "column": col,
                          "value": applicant[col], "bin": b.label,
                          "points": row["points"]})
    return {"points": breakdown, "score": int(total)}


def score_to_prob(score: float, scaling: Scaling) -> float:
    """Invert the scaling: score -> probability of going bad."""
    ln_odds_good = (score - scaling.offset) / scaling.factor
    return 1.0 / (1.0 + math.exp(ln_odds_good))


def scores_from_proba(p_bad: np.ndarray, scaling: Scaling) -> np.ndarray:
    p = np.clip(p_bad, 1e-9, 1 - 1e-9)
    return scaling.offset + scaling.factor * np.log((1 - p) / p)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling — no sklearn needed."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = np.asarray(p)[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ks_stat(y: np.ndarray, score: np.ndarray) -> float:
    """Max separation between the good and bad score distributions."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(score)
    ys = y[order]
    cum_bad = np.cumsum(ys) / max(ys.sum(), 1)
    cum_good = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def band_table(y: np.ndarray, score: np.ndarray, n_bands: int = 10,
               scaling: Scaling | None = None) -> list:
    """Bad rate by score band — the calibration check a risk committee reads."""
    edges = np.unique(np.quantile(score, np.linspace(0, 1, n_bands + 1)))
    idx = np.clip(np.digitize(score, edges[1:-1], right=True), 0,
                  len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        if not m.any():
            continue
        row = {
            "band": f"{int(round(edges[k]))}-{int(round(edges[k + 1]))}",
            "count": int(m.sum()), "bads": int(y[m].sum()),
            "actual_bad_rate": round(float(y[m].mean()), 4),
        }
        if scaling is not None:
            row["predicted_bad_rate"] = round(
                float(np.mean([score_to_prob(s, scaling) for s in score[m]])), 4)
        rows.append(row)
    return rows


def cutoff_table(y: np.ndarray, score: np.ndarray, step: int = 5) -> list:
    """Approve above the cutoff. Cost uses the dataset's official 5:1 matrix."""
    lo, hi = int(np.floor(score.min())), int(np.ceil(score.max()))
    rows = []
    for cut in range(lo, hi + 1, step):
        approved = score >= cut
        n_appr = int(approved.sum())
        bad_appr = int(y[approved].sum())            # accepted a bad -> 5
        good_rej = int((1 - y[~approved]).sum())     # turned away a good -> 1
        rows.append({
            "cutoff": cut,
            "approval_rate": round(n_appr / len(y), 4),
            "bad_rate_approved": round(bad_appr / n_appr, 4) if n_appr else 0.0,
            "bads_accepted": bad_appr, "goods_rejected": good_rej,
            "cost": round(COST_FALSE_GOOD * bad_appr
                          + COST_FALSE_BAD * good_rej, 1),
            "cost_per_applicant": round(
                (COST_FALSE_GOOD * bad_appr + COST_FALSE_BAD * good_rej)
                / len(y), 4),
        })
    return rows


def cost_at(y: np.ndarray, score: np.ndarray, cutoff: float) -> dict:
    """Cost of applying an already-chosen cutoff to a fresh population."""
    approved = score >= cutoff
    n_appr = int(approved.sum())
    bad_appr = int(y[approved].sum())
    good_rej = int((1 - y[~approved]).sum())
    total = COST_FALSE_GOOD * bad_appr + COST_FALSE_BAD * good_rej
    return {
        "cutoff": int(cutoff),
        "approval_rate": round(n_appr / len(y), 4),
        "bad_rate_approved": round(bad_appr / n_appr, 4) if n_appr else 0.0,
        "bads_accepted": bad_appr, "goods_rejected": good_rej,
        "cost": round(total, 1),
        "cost_per_applicant": round(total / len(y), 4),
    }


# --------------------------------------------------------------------------
# Challengers — is the interpretable model actually costing us anything?
# --------------------------------------------------------------------------

def challenger_models(df_tr, y_tr, df_te, y_te, features) -> list:
    """A scorecard is chosen for interpretability, not accuracy. This is the
    price of that choice, measured instead of assumed."""
    out = []
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        return out

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat = [c for c in features if c not in NUMERIC]
    num = [c for c in features if c in NUMERIC]
    Xtr = np.hstack([enc.fit_transform(df_tr[cat]),
                     df_tr[num].to_numpy(dtype=float)])
    Xte = np.hstack([enc.transform(df_te[cat]),
                     df_te[num].to_numpy(dtype=float)])

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit((Xtr - mu) / sd, y_tr)
    out.append({
        "model": "Logistic on raw one-hot (no binning)",
        "test_auc": round(float(auc_score(y_te, lr.predict_proba((Xte - mu) / sd)[:, 1])), 4),
    })

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                        max_leaf_nodes=8, random_state=0)
    gb.fit(Xtr, y_tr)
    out.append({
        "model": "Gradient boosting (black box)",
        "test_auc": round(float(auc_score(y_te, gb.predict_proba(Xte)[:, 1])), 4),
    })
    return out


# --------------------------------------------------------------------------
# Fairness audit on the characteristics the model refuses to use
# --------------------------------------------------------------------------

SEX_OF = {
    "male: divorced": "male", "male: single": "male",
    "male: married/widowed": "male",
    "female: div/sep/married": "female", "female: single": "female",
}


def fairness_audit(df: pd.DataFrame, y: np.ndarray, score: np.ndarray,
                   cutoff: int) -> dict:
    """Excluding a protected attribute does not guarantee a neutral outcome —
    other characteristics can proxy for it. Measure, don't assume."""
    groups = {
        "sex": df["personal_status_sex"].map(SEX_OF).fillna("unknown"),
        "foreign_worker": df["foreign_worker"],
    }
    out = {}
    for name, g in groups.items():
        rows = []
        for level in sorted(pd.unique(g)):
            m = (g == level).to_numpy()
            if m.sum() < 20:
                continue
            rows.append({
                "group": str(level), "n": int(m.sum()),
                "actual_bad_rate": round(float(y[m].mean()), 4),
                "approval_rate": round(float((score[m] >= cutoff).mean()), 4),
                "mean_score": round(float(score[m].mean()), 1),
                "auc": round(float(auc_score(y[m], -score[m])), 4)
                if 0 < y[m].sum() < m.sum() else None,
            })
        # A group approving nobody is the most adverse outcome there is, so it
        # belongs in the numerator — dropping zero rates would have reported a
        # comfortable ratio for the one case that most needs the flag. Only the
        # denominator has to be non-zero.
        rates = [r["approval_rate"] for r in rows]
        out[name] = {
            "rows": rows,
            # 4/5ths rule: below 0.8 is the classic adverse-impact flag
            "adverse_impact_ratio": round(min(rates) / max(rates), 4)
            if len(rates) > 1 and max(rates) > 0 else None,
        }
    return out


# --------------------------------------------------------------------------
# The whole pipeline
# --------------------------------------------------------------------------

def _pipeline(df_tr, y_tr, features, iv_floor=0.02):
    """Bin, filter by IV, select, fit — everything that learns from data."""
    binnings = bin_all(df_tr, y_tr, features)
    ranked = sorted(binnings.items(), key=lambda kv: -kv[1].iv)
    candidates = [c for c, b in ranked if b.iv >= iv_floor]
    Xw = woe_frame(df_tr, binnings, candidates)
    selected, log = select_features(Xw, y_tr, candidates)
    res = _fit_logit(Xw[selected], y_tr)
    return binnings, candidates, selected, log, res


def _predict(df, binnings, selected, res):
    Xw = woe_frame(df, binnings, selected).to_numpy(dtype=float)
    lin = res.params[0] + Xw @ np.asarray(res.params[1:], dtype=float)
    return 1.0 / (1.0 + np.exp(-lin))


def cross_validated(df, y, features, n_splits=5, n_repeats=5, seed=7):
    """Repeated stratified CV over the FULL pipeline — binning and variable
    selection included, so nothing leaks. With n=1000 a single holdout Gini
    swings by ±0.1 on the seed alone; the spread is the honest headline."""
    rng = np.random.default_rng(seed)
    yv = y.to_numpy(dtype=int)
    ginis, kss, chosen = [], [], {}
    for rep in range(n_repeats):
        folds = np.empty(len(yv), dtype=int)
        for cls in (0, 1):
            idx = np.where(yv == cls)[0]
            rng.shuffle(idx)
            folds[idx] = np.arange(len(idx)) % n_splits
        for k in range(n_splits):
            tr, te = folds != k, folds == k
            try:
                binnings, _, selected, _, res = _pipeline(
                    df[tr], y[tr], features)
                p = _predict(df[te], binnings, selected, res)
            except Exception:
                continue
            a = auc_score(yv[te], p)
            ginis.append(2 * a - 1)
            kss.append(ks_stat(yv[te], -p))
            for c in selected:
                chosen[c] = chosen.get(c, 0) + 1
    n = max(len(ginis), 1)
    return {
        "n_fits": len(ginis), "n_splits": n_splits, "n_repeats": n_repeats,
        "gini_mean": round(float(np.mean(ginis)), 4),
        "gini_std": round(float(np.std(ginis)), 4),
        "gini_p05": round(float(np.percentile(ginis, 5)), 4),
        "gini_p95": round(float(np.percentile(ginis, 95)), 4),
        "ks_mean": round(float(np.mean(kss)), 4),
        "ks_std": round(float(np.std(kss)), 4),
        "selection_frequency": {c: round(v / n, 3) for c, v in
                                sorted(chosen.items(), key=lambda kv: -kv[1])},
    }


def stratified_split(y: pd.Series, test_frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    yv = y.to_numpy(dtype=int)
    test = np.zeros(len(yv), dtype=bool)
    for cls in (0, 1):
        idx = np.where(yv == cls)[0]
        rng.shuffle(idx)
        test[idx[: int(round(test_frac * len(idx)))]] = True
    return ~test, test


def analyze(pdo: float = 20.0, base_score: float = 600.0,
            base_odds: float = 50.0, seed: int = 42,
            n_repeats: int = 5, run_cv: bool = True) -> dict:
    """Run the full scorecard build and return everything as plain JSON."""
    scaling = Scaling(pdo=pdo, base_score=base_score, base_odds=base_odds)
    df = load_data()
    y = df["target"]
    features = [c for c in COLUMNS
                if c != "target" and c not in PROTECTED]

    tr, te = stratified_split(y, seed=seed)
    df_tr, y_tr = df[tr].reset_index(drop=True), y[tr].reset_index(drop=True)
    df_te, y_te = df[te].reset_index(drop=True), y[te].reset_index(drop=True)

    # Stability selection, run INSIDE the training set only. On 700 rows a
    # forward stepwise will happily admit a variable that a different sample
    # would never pick; anything that survives fewer than half the resamples
    # is noise dressed as a characteristic.
    stability = (cross_validated(df_tr, y_tr, features, n_repeats=n_repeats,
                                 seed=seed) if run_cv else None)

    binnings, candidates, selected_raw, steps, _ = _pipeline(
        df_tr, y_tr, features)
    dropped = []
    selected = list(selected_raw)
    if stability is not None:
        freq = stability["selection_frequency"]
        dropped = [{"column": c, "pretty": PRETTY.get(c, c),
                    "frequency": freq.get(c, 0.0)}
                   for c in selected_raw if freq.get(c, 0.0) < 0.5]
        selected = [c for c in selected_raw if freq.get(c, 0.0) >= 0.5]
    res = _fit_logit(woe_frame(df_tr, binnings, selected), y_tr)
    card = build_scorecard(binnings, selected, res.params, scaling)

    p_tr = _predict(df_tr, binnings, selected, res)
    p_te = _predict(df_te, binnings, selected, res)
    s_tr = scores_from_proba(p_tr, scaling)
    s_te = scores_from_proba(p_te, scaling)
    ytr_v, yte_v = y_tr.to_numpy(dtype=int), y_te.to_numpy(dtype=int)

    # The cutoff is a decision, and it has to be made without seeing the
    # holdout: pick it on train, then pay for it on test. Choosing the
    # cost-minimising cutoff on the test set flatters the result.
    cuts_tr = cutoff_table(ytr_v, s_tr)
    best_tr = min(cuts_tr, key=lambda r: r["cost_per_applicant"])
    cuts = cutoff_table(yte_v, s_te)
    best_cut = cost_at(yte_v, s_te, best_tr["cutoff"])
    oracle_cut = min(cuts, key=lambda r: r["cost_per_applicant"])
    approve_all = {
        "strategy": "approve everyone",
        "approval_rate": 1.0,
        "cost": round(COST_FALSE_GOOD * int(yte_v.sum()), 1),
        "cost_per_applicant": round(
            COST_FALSE_GOOD * int(yte_v.sum()) / len(yte_v), 4),
    }
    approve_none = {
        "strategy": "approve nobody",
        "approval_rate": 0.0,
        "cost": round(COST_FALSE_BAD * int((1 - yte_v).sum()), 1),
        "cost_per_applicant": round(
            COST_FALSE_BAD * int((1 - yte_v).sum()) / len(yte_v), 4),
    }

    # A protected attribute's IV, for the record: it IS predictive, and it is
    # still excluded. That tension is the point.
    excluded_iv = {}
    protected_binnings = {}
    for col in PROTECTED:
        b = bin_categorical(df_tr[col], y_tr)
        protected_binnings[col] = b
        excluded_iv[col] = {"pretty": PRETTY[col], "iv": round(b.iv, 4),
                            "strength": iv_strength(b.iv)}

    def _bins_json(col, b, protected):
        return {
            "pretty": PRETTY.get(col, col), "kind": b.kind,
            "iv": round(b.iv, 4), "strength": iv_strength(b.iv),
            "protected": protected, "in_model": col in selected,
            "bins": [{"label": x.label, "count": x.count, "bads": x.bads,
                      "bad_rate": round(x.bad_rate, 4),
                      "woe": round(x.woe, 4), "iv": round(x.iv, 4)}
                     for x in b.bins],
        }

    binning_all = {c: _bins_json(c, binnings[c], False) for c in features}
    binning_all.update({c: _bins_json(c, protected_binnings[c], True)
                        for c in PROTECTED})

    out = {
        "meta": {
            "n_total": int(len(df)), "n_train": int(len(df_tr)),
            "n_test": int(len(df_te)),
            "bad_rate": round(float(y.mean()), 4),
            "pdo": pdo, "base_score": base_score, "base_odds": base_odds,
            "factor": round(scaling.factor, 4),
            "offset": round(scaling.offset, 4),
            "seed": seed,
            "cost_false_good": COST_FALSE_GOOD,
            "cost_false_bad": COST_FALSE_BAD,
        },
        "iv_ranking": [
            {"column": c, "pretty": PRETTY.get(c, c),
             "iv": round(binnings[c].iv, 4),
             "strength": iv_strength(binnings[c].iv),
             "n_bins": len(binnings[c].bins),
             "in_model": c in selected}
            for c in sorted(features, key=lambda c: -binnings[c].iv)
        ],
        "excluded_protected": excluded_iv,
        "binning_all": binning_all,
        "iv_floor": 0.02,
        "candidates": candidates,
        "selection_steps": steps,
        "selected_by_stepwise": selected_raw,
        "dropped_for_instability": dropped,
        "stability_threshold": 0.5,
        "selected": selected,
        "model": {
            "intercept": round(float(res.params[0]), 4),
            "pseudo_r2": round(float(res.prsquared), 4),
            "llr_p_value": float(res.llr_pvalue),
            "terms": [
                {"column": c, "pretty": PRETTY.get(c, c),
                 "coefficient": round(float(res.params[i + 1]), 4),
                 "std_err": round(float(res.bse[i + 1]), 4),
                 "p_value": float(res.pvalues[i + 1])}
                for i, c in enumerate(selected)
            ],
        },
        "scorecard": card,
        "score_range": {
            "min": int(sum(min(r["points"] for r in e["rows"])
                           for e in card.values())),
            "max": int(sum(max(r["points"] for r in e["rows"])
                           for e in card.values())),
        },
        "performance": {
            "train": {"auc": round(float(auc_score(ytr_v, p_tr)), 4),
                      "gini": round(float(2 * auc_score(ytr_v, p_tr) - 1), 4),
                      "ks": round(float(ks_stat(ytr_v, s_tr)), 4)},
            "test": {"auc": round(float(auc_score(yte_v, p_te)), 4),
                     "gini": round(float(2 * auc_score(yte_v, p_te) - 1), 4),
                     "ks": round(float(ks_stat(yte_v, s_te)), 4)},
            "psi_train_vs_test": round(psi(s_tr, s_te), 4),
        },
        "calibration": band_table(yte_v, s_te, scaling=scaling),
        "cutoffs": cuts,
        "cutoffs_train": cuts_tr,
        "cutoff_chosen_on_train": best_tr,
        "best_cutoff": best_cut,
        "oracle_cutoff_test": oracle_cut,
        "cost_baselines": [approve_all, approve_none],
        "challengers": challenger_models(df_tr, ytr_v, df_te, yte_v, features),
        "fairness": fairness_audit(df_te, yte_v, s_te, best_cut["cutoff"]),
        "score_distribution": {
            "goods": [round(float(v), 1) for v in s_te[yte_v == 0]],
            "bads": [round(float(v), 1) for v in s_te[yte_v == 1]],
        },
        # Everything the front end needs to score an applicant on its own:
        # bin boundaries / level sets plus the points attached to each bin.
        "binning_spec": {
            c: {
                "pretty": PRETTY.get(c, c),
                "kind": binnings[c].kind,
                "bins": [
                    {"label": b.label,
                     "lo": (None if b.lo == -np.inf else float(b.lo)),
                     "hi": (None if b.hi == np.inf else float(b.hi)),
                     "levels": [str(l) for l in b.levels],
                     "woe": round(b.woe, 4),
                     "bad_rate": round(b.bad_rate, 4),
                     "count": b.count,
                     "points": row["points"]}
                    for b, row in zip(binnings[c].bins, card[c]["rows"])
                ],
            }
            for c in selected
        },
        "bin_options": {
            c: ([b.label for b in binnings[c].bins] if c not in NUMERIC
                else [b.label for b in binnings[c].bins])
            for c in selected
        },
        "raw_levels": {c: sorted(pd.unique(df[c]).tolist())
                       for c in selected if c not in NUMERIC},
        "numeric_ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()),
                "median": float(df[c].median())}
            for c in selected if c in NUMERIC
        },
    }
    if stability is not None:
        out["cross_validation"] = stability
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _table(rows: list, cols: list, headers: list) -> str:
    data = [[f"{r.get(c, '')}" for c in cols] for r in rows]
    widths = [max(len(h), *(len(d[i]) for d in data)) if data else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for d in data:
        out.append("  ".join(v.ljust(w) for v, w in zip(d, widths)))
    return "\n".join(out)


def make_chart(res: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cut = res["best_cutoff"]["cutoff"]
    goods = np.array(res["score_distribution"]["goods"])
    bads = np.array(res["score_distribution"]["bads"])

    edges = np.linspace(min(goods.min(), bads.min()),
                        max(goods.max(), bads.max()), 22)
    ax[0, 0].hist(goods, bins=edges, alpha=.65, label="good", color="#2b6cb0",
                  density=True)
    ax[0, 0].hist(bads, bins=edges, alpha=.65, label="bad", color="#c53030",
                  density=True)
    ax[0, 0].axvline(cut, color="k", ls="--", lw=1,
                     label=f"cutoff {cut}")
    ax[0, 0].set_title("Score distribution (holdout)")
    ax[0, 0].set_xlabel("score")
    ax[0, 0].margins(y=0.14)
    ax[0, 0].legend()

    alls = np.concatenate([goods, bads])
    y = np.concatenate([np.zeros(len(goods)), np.ones(len(bads))])
    grid = np.linspace(alls.min(), alls.max(), 200)
    cum_bad = [(bads <= t).mean() for t in grid]
    cum_good = [(goods <= t).mean() for t in grid]
    k = int(np.argmax(np.abs(np.array(cum_bad) - np.array(cum_good))))
    ax[0, 1].plot(grid, cum_bad, color="#c53030", label="cumulative bad")
    ax[0, 1].plot(grid, cum_good, color="#2b6cb0", label="cumulative good")
    ax[0, 1].vlines(grid[k], cum_good[k], cum_bad[k], color="k", lw=2)
    ax[0, 1].set_title(f"KS = {res['performance']['test']['ks']:.3f}")
    ax[0, 1].set_xlabel("score")
    ax[0, 1].legend()

    cal = res["calibration"]
    xs = np.arange(len(cal))
    ax[1, 0].bar(xs - .2, [r["actual_bad_rate"] for r in cal], .4,
                 label="actual", color="#c53030")
    ax[1, 0].bar(xs + .2, [r["predicted_bad_rate"] for r in cal], .4,
                 label="predicted", color="#718096")
    ax[1, 0].set_xticks(xs)
    ax[1, 0].set_xticklabels([r["band"] for r in cal], rotation=45, ha="right",
                             fontsize=7)
    ax[1, 0].set_title("Calibration: bad rate by score band (holdout)")
    ax[1, 0].legend()

    cuts = res["cutoffs"]
    ax[1, 1].plot([c["cutoff"] for c in cuts],
                  [c["cost_per_applicant"] for c in cuts],
                  color="#2b6cb0", label="scorecard policy")
    for b, color in zip(res["cost_baselines"], ["#c53030", "#dd6b20"]):
        ax[1, 1].axhline(b["cost_per_applicant"], ls=":", color=color,
                         label=b["strategy"])
    ax[1, 1].axvline(cut, color="k", ls="--", lw=1, label=f"chosen {cut}")
    ax[1, 1].set_title("Expected cost per applicant (5:1 cost matrix)")
    ax[1, 1].set_xlabel("cutoff")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main() -> None:
    res = analyze()
    m = res["meta"]
    print("=" * 78)
    print("CREDIT SCORECARD — German Credit (Statlog), "
          f"n={m['n_total']}, bad rate {m['bad_rate']:.1%}")
    print(f"train {m['n_train']} / holdout {m['n_test']}  |  "
          f"PDO {m['pdo']:.0f}, {m['base_score']:.0f} pts at "
          f"{m['base_odds']:.0f}:1 odds")
    print("=" * 78)

    print("\n1. INFORMATION VALUE (train, after coarse classing)\n")
    print(_table(res["iv_ranking"],
                 ["pretty", "n_bins", "iv", "strength", "in_model"],
                 ["characteristic", "bins", "IV", "strength", "in model"]))
    print("\n   Excluded by policy (protected characteristics):")
    for col, e in res["excluded_protected"].items():
        print(f"     {e['pretty']:<32} IV {e['iv']:.4f}  ({e['strength']})")

    print("\n2. MODEL — logistic regression on WOE\n")
    if res.get("dropped_for_instability"):
        for d in res["dropped_for_instability"]:
            print(f"   dropped by stability selection: {d['pretty']} "
                  f"(entered the stepwise, survived only "
                  f"{d['frequency']:.0%} of resamples)")
        print()
    print(_table(res["model"]["terms"],
                 ["pretty", "coefficient", "std_err", "p_value"],
                 ["characteristic", "coef", "s.e.", "p-value"]))
    print(f"\n   intercept {res['model']['intercept']:+.4f}   "
          f"pseudo-R2 {res['model']['pseudo_r2']:.4f}")

    print("\n3. SCORECARD (points)\n")
    for col, entry in res["scorecard"].items():
        print(f"   {entry['pretty']}  (IV {entry['iv']:.3f})")
        print("   " + _table(entry["rows"],
                             ["bin", "count", "bad_rate", "woe", "points"],
                             ["bin", "n", "bad rate", "WOE", "points"]
                             ).replace("\n", "\n   "))
        print()
    print(f"   Score range: {res['score_range']['min']} to "
          f"{res['score_range']['max']}")

    print("\n4. DISCRIMINATION\n")
    p = res["performance"]
    print(f"   train   AUC {p['train']['auc']:.4f}  Gini "
          f"{p['train']['gini']:.4f}  KS {p['train']['ks']:.4f}")
    print(f"   holdout AUC {p['test']['auc']:.4f}  Gini "
          f"{p['test']['gini']:.4f}  KS {p['test']['ks']:.4f}")
    print(f"   PSI train vs holdout: {p['psi_train_vs_test']:.4f} "
          "(<0.10 = stable)")
    if "cross_validation" in res:
        cv = res["cross_validation"]
        print(f"\n   Repeated CV inside train ({cv['n_repeats']}x"
              f"{cv['n_splits']}, full pipeline re-run per fold, "
              f"{cv['n_fits']} fits):")
        print(f"     Gini {cv['gini_mean']:.4f} +/- {cv['gini_std']:.4f}   "
              f"90% range [{cv['gini_p05']:.4f}, {cv['gini_p95']:.4f}]")
        print(f"     KS   {cv['ks_mean']:.4f} +/- {cv['ks_std']:.4f}")
        print("     variable selection frequency across folds:")
        for c, f in cv["selection_frequency"].items():
            print(f"       {PRETTY.get(c, c):<34} {f:.0%}")

    print("\n5. CHALLENGERS (holdout AUC)\n")
    print(f"   {'Scorecard (WOE + logistic)':<40} "
          f"{p['test']['auc']:.4f}")
    for c in res["challengers"]:
        print(f"   {c['model']:<40} {c['test_auc']:.4f}")

    print("\n6. CUTOFF — official 5:1 cost matrix\n")
    b, o = res["best_cutoff"], res["oracle_cutoff_test"]
    t = res["cutoff_chosen_on_train"]
    print(f"   chosen on train: {t['cutoff']} "
          f"(train cost/applicant {t['cost_per_applicant']:.3f})")
    print(f"   applied to holdout: approve {b['approval_rate']:.1%}, "
          f"bad rate among approved {b['bad_rate_approved']:.1%}, "
          f"cost/applicant {b['cost_per_applicant']:.3f}")
    for base in res["cost_baselines"]:
        print(f"     baseline {base['strategy']:<20} "
              f"cost/applicant {base['cost_per_applicant']:.3f}")
    print(f"     (an oracle picking the cutoff on the holdout itself would "
          f"get {o['cost_per_applicant']:.3f} at {o['cutoff']} — "
          "that number is not honest, it is the size of the temptation)")

    print("\n7. CALIBRATION — bad rate by score band (holdout)\n")
    print(_table(res["calibration"],
                 ["band", "count", "actual_bad_rate", "predicted_bad_rate"],
                 ["score band", "n", "actual bad", "predicted bad"]))

    print("\n8. FAIRNESS — groups the model is not allowed to score on\n")
    for name, block in res["fairness"].items():
        print(f"   by {name}:")
        print("   " + _table(block["rows"],
                             ["group", "n", "actual_bad_rate",
                              "approval_rate", "mean_score", "auc"],
                             ["group", "n", "bad rate", "approval",
                              "mean score", "AUC"]).replace("\n", "\n   "))
        air = block["adverse_impact_ratio"]
        if air is not None:
            flag = "OK" if air >= 0.8 else "BELOW the 4/5ths threshold"
            print(f"     adverse impact ratio {air:.3f} -> {flag}\n")

    path = make_chart(res, os.path.join(HERE, "scorecard.png"))
    print(f"\nChart written to {path}")

    with open(os.path.join(HERE, "scorecard_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("Full results written to scorecard_results.json")


if __name__ == "__main__":
    main()
