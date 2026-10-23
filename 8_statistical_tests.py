"""

Input   ./generated_sweep/sweep_run_level.csv
Output  ./results/


Metrics
--------------------------------------------------------------------------
  primary    yield              Higher         novel+unique+valid molecules per 10,000
  primary    genuine_yield      Higher         ... of those, nearest-seed Tanimoto < 0.7
  secondary  validity           Higher
  secondary  uniqueness         Higher
  secondary  novelty            Higher
  secondary  seed_recovery      Lower          fraction of the 1,000 seeds regenerated

Usage:
    python3 significance_tests.py
    python3 significance_tests.py --csv <path> --out <dir>
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "generated_sweep", "sweep_run_level.csv")
DEFAULT_OUT = os.path.join(HERE, "results")

# --------------------------------------------------------------------------
# Experiment 
# --------------------------------------------------------------------------
FAMILIES = {
    "family_A": {
        "label": "Family A - corpus size: does more augmentation help?",
        "arms": ["part1_arm_a", "part1_arm_b", "part1_arm_c"],
        "pretty": {"part1_arm_a": "1K", "part1_arm_b": "25K", "part1_arm_c": "50K"},
    },
    "family_B": {
        "label": "Family B - randomization axis: which axis drives the gain?",
        "arms": ["part2_arm_atom", "part2_arm_dot", "part2_arm_full"],
        "pretty": {"part2_arm_atom": "atom", "part2_arm_dot": "dot",
                   "part2_arm_full": "full"},
    },
    "family_C": {
        "label": "Family C - representation: does SAFE beat randomized SMILES?",
        "arms": ["part3_arm_safe", "part3_arm_smiles"],
        "pretty": {"part3_arm_safe": "SAFE", "part3_arm_smiles": "SMILES"},
    },
}

# canonical name -> (column in all_arms_metrics.csv, role, direction)
# "up" = higher is better. Column names verified against the real file.
METRICS = {
    "yield":         ("yield",         "primary",   "up"),
    "genuine_yield": ("genuine_yield", "primary",   "up"),
    "validity":      ("validity",      "secondary", "up"),
    "uniqueness":    ("uniqueness",    "secondary", "up"),
    "novelty":       ("novelty",       "secondary", "up"),
    "seed_recovery": ("seed_recovery", "secondary", "down"),
    # "near_dup_frac": ("near_dup_frac", "secondary", "down"),
    # "nn_median":     ("nn_median",     "secondary", "down"),
}
PRIMARY = [m for m, (_, r, _) in METRICS.items() if r == "primary"]
ROLE = {m: r for m, (_, r, _) in METRICS.items()}
DIRECTION = {m: d for m, (_, _, d) in METRICS.items()}
COLUMN = {m: c for m, (c, _, _) in METRICS.items()}

EFFECT_SIZE_NOTE = ("classical equal-variance SS decomposition; descriptive "
                    "companion to the Welch test, not a Welch-specific effect size")

ARM_COL = "arm"
SET_COL = "train_seed"
N_EXPECTED = 5
# descriptive only 
DIVERSITY = ["intdiv_novel_mol", "intdiv_novel_scaffold"]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def welch_anova(groups):
    """Welch one-way ANOVA for k groups with unequal variances."""
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])

    if np.any(v <= 0):
        return dict(welch_F=np.nan, df1=k - 1, df2=np.nan, welch_p=np.nan,
                    eta2_classical_descriptive=np.nan,
                    omega2_classical_descriptive=np.nan,
                    effect_size_note=EFFECT_SIZE_NOTE,
                    note="a group has zero variance; Welch is undefined")

    w = n / v                       # per-group precision weight
    W = w.sum()
    m_w = (w * m).sum() / W         # weighted grand mean

    lam = ((1 - w / W) ** 2 / (n - 1)).sum()
    A = (w * (m - m_w) ** 2).sum() / (k - 1)
    B = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lam

    F = A / B
    df1 = k - 1
    df2 = 1.0 / ((3.0 / (k ** 2 - 1)) * lam)
    p = stats.f.sf(F, df1, df2)

    # Variance-explained from the CLASSICAL equal-variance sums-of-squares
    # decomposition. Welch's ANOVA has no single agreed-upon variance-explained
    # statistic, so these are descriptive companions to the Welch test, NOT
    # "the effect size of the Welch ANOVA". They are named and annotated in the
    # output so they cannot be read that way.
    allx = np.concatenate(groups)
    grand = allx.mean()
    ss_b = (n * (m - grand) ** 2).sum()
    ss_w = sum(((np.asarray(g) - gm) ** 2).sum() for g, gm in zip(groups, m))
    ss_t = ss_b + ss_w
    ms_w = ss_w / (allx.size - k)
    eta2 = ss_b / ss_t if ss_t > 0 else np.nan
    omega2 = (ss_b - (k - 1) * ms_w) / (ss_t + ms_w) if (ss_t + ms_w) > 0 else np.nan
    return dict(welch_F=F, df1=df1, df2=df2, welch_p=p,
                eta2_classical_descriptive=eta2,
                omega2_classical_descriptive=omega2,
                effect_size_note=EFFECT_SIZE_NOTE, note="")


def _solve_ncp(t_obs, df, target_cdf):
    """Noncentrality lambda with nct.cdf(t_obs, df, lambda) == target_cdf. """
    def f(lam):
        try:
            v = stats.nct.cdf(t_obs, df, lam)
        except Exception:
            return np.nan
        return v - target_cdf if np.isfinite(v) else np.nan

    if not np.isfinite(t_obs):
        return np.nan
    lo = hi = float(t_obs)
    flo = fhi = f(lo)
    if not np.isfinite(flo):
        return np.nan
    if flo == 0.0:
        return lo

    step = max(1.0, abs(t_obs) * 0.25)
    floor = 1e-9 * max(1.0, abs(t_obs))
    go_right = flo > 0          # cdf decreases in lam, so a positive f needs a larger lam

    for _ in range(400):
        nxt = (hi + step) if go_right else (lo - step)
        v = f(nxt)
        if not np.isfinite(v):          # overshot into the underflow region
            step *= 0.5
            if step < floor:
                return np.nan
            continue
        if go_right:
            lo, flo, hi, fhi = hi, fhi, nxt, v
            if fhi <= 0:
                break
        else:
            hi, fhi, lo, flo = lo, flo, nxt, v
            if flo >= 0:
                break
        step *= 2.0
    else:
        return np.nan

    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return np.nan
    try:
        return float(optimize.brentq(f, lo, hi, maxiter=200))
    except Exception:
        return np.nan


def hedges_g(a, b, alpha=0.05):
    """Hedges' g (bias-corrected Cohen's d) with an exact noncentral-t 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s_pool = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    J = 1 - 3.0 / (4 * (na + nb) - 9)               # small-sample correction

    if not np.isfinite(s_pool) or s_pool <= 0:
        return dict(hedges_g=np.nan, g_se=np.nan, g_ci_low=np.nan,
                    g_ci_high=np.nan, g_ci_method="undefined (zero pooled SD)")

    d = (a.mean() - b.mean()) / s_pool
    g = J * d
    se_d = np.sqrt((na + nb) / (na * nb) + d ** 2 / (2 * (na + nb)))
    se_g = J * se_d

    df = na + nb - 2
    c = np.sqrt(na * nb / (na + nb))                # d -> t scaling
    t_obs = d * c
    lam_lo = _solve_ncp(t_obs, df, 1 - alpha / 2)   # lower limit
    lam_hi = _solve_ncp(t_obs, df, alpha / 2)       # upper limit

    if np.isfinite(lam_lo) and np.isfinite(lam_hi):
        return dict(hedges_g=g, g_se=se_g,
                    g_ci_low=J * lam_lo / c, g_ci_high=J * lam_hi / c,
                    g_ci_method="noncentral-t")

    z = stats.norm.ppf(1 - alpha / 2)               # last resort only
    return dict(hedges_g=g, g_se=se_g,
                g_ci_low=g - z * se_g, g_ci_high=g + z * se_g,
                g_ci_method="normal-approx (noncentral-t did not converge)")


def games_howell(groups, names, alpha=0.05):
    """Games-Howell post-hoc: all pairs, unequal variances, unequal n."""
    k = len(groups)
    out = []
    for i, j in itertools.combinations(range(k), 2):
        a, b = np.asarray(groups[i], float), np.asarray(groups[j], float)
        na, nb = a.size, b.size
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)

        se2 = va / na + vb / nb                     # variance of the difference
        se = np.sqrt(se2)
        diff = ma - mb
        if se2 <= 0:
            out.append(dict(group_A=names[i], group_B=names[j],
                            mean_A=ma, sd_A=np.sqrt(va), n_A=na,
                            mean_B=mb, sd_B=np.sqrt(vb), n_B=nb,
                            mean_diff=diff, ci_low=np.nan, ci_high=np.nan,
                            se=0.0, t=np.nan, q=np.nan, df=np.nan, p_value=np.nan,
                            **hedges_g(a, b)))
            continue
        # Welch-Satterthwaite df for this pair
        df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        t = diff / se
        q = np.sqrt(2) * abs(t)                     # studentized range statistic
        p = float(stats.studentized_range.sf(q, k, df))
        q_crit = stats.studentized_range.ppf(1 - alpha, k, df)
        half = q_crit / np.sqrt(2) * se
        out.append(dict(
            group_A=names[i], group_B=names[j],
            mean_A=ma, sd_A=np.sqrt(va), n_A=na,
            mean_B=mb, sd_B=np.sqrt(vb), n_B=nb,
            mean_diff=diff, ci_low=diff - half, ci_high=diff + half,
            se=se, t=t, q=q, df=df, p_value=min(p, 1.0),
            **hedges_g(a, b),
        ))
    return out


def welch_ttest(a, b, alpha=0.05):
    """Welch two-sided t-test plus the CI of the mean difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    diff = a.mean() - b.mean()
    if se <= 0:
        return dict(mean_A=a.mean(), sd_A=np.sqrt(va), n_A=na,
                    mean_B=b.mean(), sd_B=np.sqrt(vb), n_B=nb,
                    mean_diff=diff, ci_low=np.nan, ci_high=np.nan,
                    se=0.0, t=np.nan, df=np.nan, p_value=np.nan)
    df = se ** 4 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t, p = stats.ttest_ind(a, b, equal_var=False)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return dict(mean_A=a.mean(), sd_A=np.sqrt(va), n_A=na,
                mean_B=b.mean(), sd_B=np.sqrt(vb), n_B=nb,
                mean_diff=diff, ci_low=diff - tcrit * se, ci_high=diff + tcrit * se,
                se=se, t=float(t), df=df, p_value=float(p))


def levene_median(groups):
    """Levene's test, median-centered (Brown-Forsythe): robust to non-normality."""
    try:
        W, p = stats.levene(*groups, center="median")
        return dict(levene_W=float(W), levene_p=float(p))
    except Exception as e:
        return dict(levene_W=np.nan, levene_p=np.nan, levene_note=str(e))


def sd_ratio(groups):
    sds = [float(np.std(g, ddof=1)) for g in groups]
    lo = min(sds)
    return max(sds) / lo if lo > 0 else np.inf


def better_arm(name_a, name_b, mean_a, mean_b, direction):
    if np.isclose(mean_a, mean_b):
        return "tie"
    if direction == "up":
        return name_a if mean_a > mean_b else name_b
    return name_a if mean_a < mean_b else name_b


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def validate(df):
    need = [ARM_COL, SET_COL] + [COLUMN[m] for m in METRICS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"[schema] missing required columns: {missing}\n"
                 f"         available: {list(df.columns)}")

    counts = df[ARM_COL].value_counts().to_dict()
    expected = [a for f in FAMILIES.values() for a in f["arms"]]
    problems = [f"  {a}: found {counts.get(a, 0)} rows, expected {N_EXPECTED}"
                for a in expected if counts.get(a, 0) != N_EXPECTED]
    if problems:
        sys.exit("[design] generation-set counts are wrong - refusing to run statistics:\n"
                 + "\n".join(problems)
                 + f"\n  arms present: {sorted(counts)}")

    bad = []
    for a in expected:
        for m in METRICS:
            v = df.loc[df[ARM_COL] == a, COLUMN[m]]
            if v.isna().any():
                bad.append(f"  {a}/{m}: {int(v.isna().sum())} NaN")
    if bad:
        sys.exit("[data] NaNs in metrics used for testing:\n" + "\n".join(bad))

    extra = sorted(set(counts) - set(expected))
    if extra:
        print(f"[design] note: arms present but not in the design, ignored: {extra}")
    return [c for c in DIVERSITY if c in df.columns]


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def strip_plot(df, fam_key, fam, metric, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = COLUMN[metric]
    arms = fam["arms"]
    labels = [fam["pretty"][a] for a in arms]
    data = [df.loc[df[ARM_COL] == a, col].to_numpy(float) for a in arms]

    fig, ax = plt.subplots(figsize=(1.9 * len(arms) + 1.8, 4.4))
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        x = i + rng.uniform(-0.07, 0.07, vals.size)   # tiny jitter; all 5 stay visible
        ax.scatter(x, vals, s=52, zorder=3, edgecolor="white", linewidth=0.8)
        m, sd = vals.mean(), vals.std(ddof=1)
        ax.hlines(m, i - 0.26, i + 0.26, color="black", linewidth=1.9, zorder=4)
        ax.vlines(i, m - sd, m + sd, color="black", linewidth=1.1, zorder=2)

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    arrow = "higher is better" if DIRECTION[metric] == "up" else "LOWER is better"
    ax.set_ylabel(f"{metric}  ({arrow})")
    ax.set_title(f"{fam_key.replace('_', ' ')}: {metric}\n"
                 f"each point = for one train set, average of three 10,000-molecule generation sets (n={N_EXPECTED}); "
                 f"bar = mean, whisker = SD", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    csv = os.path.abspath(args.csv)
    out = os.path.abspath(args.out)

    if not os.path.isfile(csv):
        sys.exit(f"[input] not found: {csv}")

    df = pd.read_csv(csv)
    div_cols = validate(df)

    print(f"[input] {csv}")
    print(
        f"        {len(df)} rows, {df[ARM_COL].nunique()} arms, "
        f"{N_EXPECTED} training sets each"
    )

    os.makedirs(out, exist_ok=True)

    for fam_key, fam in FAMILIES.items():
        fdir = os.path.join(out, fam_key)
        os.makedirs(fdir, exist_ok=True)

        arms, pretty = fam["arms"], fam["pretty"]

        grp = {
            m: [
                df.loc[df[ARM_COL] == a, COLUMN[m]].to_numpy(float)
                for a in arms
            ]
            for m in METRICS
        }

        rows = []
        for m in METRICS:
            for a, v in zip(arms, grp[m]):
                rows.append(
                    dict(
                        metric=m,
                        role=ROLE[m],
                        direction=DIRECTION[m],
                        arm=a,
                        arm_label=pretty[a],
                        n=v.size,
                        mean=v.mean(),
                        sd=v.std(ddof=1),
                        min=v.min(),
                        max=v.max(),
                    )
                )

        pd.DataFrame(rows).to_csv(
            os.path.join(fdir, "descriptives.csv"),
            index=False,
        )

        lev = pd.DataFrame([
            dict(
                metric=m,
                role=ROLE[m],
                direction=DIRECTION[m],
                **levene_median(grp[m]),
                sd_ratio_max_min=sd_ratio(grp[m]),
            )
            for m in METRICS
        ])

        lev.to_csv(
            os.path.join(fdir, "levene.csv"),
            index=False,
        )

        lp = lev.set_index("metric")

        if len(arms) == 3:
            an = pd.DataFrame([
                dict(
                    metric=m,
                    role=ROLE[m],
                    direction=DIRECTION[m],
                    **welch_anova(grp[m]),
                )
                for m in METRICS
            ])

            an.to_csv(
                os.path.join(fdir, "welch_anova.csv"),
                index=False,
            )

            gh = []

            for m in METRICS:
                for r in games_howell(
                    grp[m],
                    [pretty[a] for a in arms],
                ):
                    r.update(
                        metric=m,
                        role=ROLE[m],
                        direction=DIRECTION[m],
                    )

                    r["better_arm"] = better_arm(
                        r["group_A"],
                        r["group_B"],
                        r["mean_A"],
                        r["mean_B"],
                        DIRECTION[m],
                    )

                    r["ci_excludes_zero"] = bool(
                        r["ci_low"] > 0 or r["ci_high"] < 0
                    )

                    gh.append(r)

            gh = pd.DataFrame(gh)

            front = [
                "metric",
                "role",
                "direction",
                "group_A",
                "group_B",
                "better_arm",
                "ci_excludes_zero",
            ]

            gh = gh[
                front + [c for c in gh.columns if c not in front]
            ]

            gh.to_csv(
                os.path.join(fdir, "games_howell.csv"),
                index=False,
            )

        else:
            a_name, b_name = arms
            tt, hg = [], []

            for m in METRICS:
                a, b = grp[m]

                r = welch_ttest(a, b)
                r.update(
                    metric=m,
                    role=ROLE[m],
                    direction=DIRECTION[m],
                    group_A=pretty[a_name],
                    group_B=pretty[b_name],
                )

                r["better_arm"] = better_arm(
                    pretty[a_name],
                    pretty[b_name],
                    r["mean_A"],
                    r["mean_B"],
                    DIRECTION[m],
                )

                r["ci_excludes_zero"] = bool(
                    r["ci_low"] > 0 or r["ci_high"] < 0
                )

                tt.append(r)

                g = hedges_g(a, b)
                g.update(
                    metric=m,
                    role=ROLE[m],
                    direction=DIRECTION[m],
                    group_A=pretty[a_name],
                    group_B=pretty[b_name],
                )

                hg.append(g)

            tt = pd.DataFrame(tt)

            front = [
                "metric",
                "role",
                "direction",
                "group_A",
                "group_B",
                "better_arm",
                "ci_excludes_zero",
            ]

            tt = tt[
                front + [c for c in tt.columns if c not in front]
            ]

            tt.to_csv(
                os.path.join(fdir, "welch_ttest.csv"),
                index=False,
            )

            hg = pd.DataFrame(hg)

            hg = hg[
                [
                    "metric",
                    "role",
                    "direction",
                    "group_A",
                    "group_B",
                    "hedges_g",
                    "g_ci_low",
                    "g_ci_high",
                    "g_ci_method",
                    "g_se",
                ]
            ]

            hg.to_csv(
                os.path.join(fdir, "hedges_g.csv"),
                index=False,
            )

        for m in PRIMARY:
            pass
            # strip_plot(
            #     df,
            #     fam_key,
            #     fam,
            #     m,
            #     os.path.join(fdir, f"{m}_stripplot.png"),
            # )

    if div_cols:
        rows = []

        for fam_key, fam in FAMILIES.items():
            for col in div_cols:
                means = []

                for a in fam["arms"]:
                    v = df.loc[
                        df[ARM_COL] == a, col
                    ].to_numpy(float)

                    means.append(v.mean())

                    rows.append(
                        dict(
                            family=fam_key,
                            metric=col,
                            arm=a,
                            arm_label=fam["pretty"][a],
                            n=v.size,
                            mean=v.mean(),
                            sd=v.std(ddof=1),
                            min=v.min(),
                            max=v.max(),
                        )
                    )

                means = np.array(means)

                rows.append(
                    dict(
                        family=fam_key,
                        metric=col,
                        arm="ACROSS-ARM RANGE OF MEANS",
                        arm_label="",
                        n=means.size,
                        mean=means.mean(),
                        sd=np.nan,
                        min=means.min(),
                        max=means.max(),
                    )
                )

        pd.DataFrame(rows).to_csv(
            os.path.join(out, "internal_diversity_descriptive.csv"),
            index=False,
        )

    with open(
        os.path.join(out, "analysis_config.json"), "w"
    ) as f:
        json.dump(
            dict(
                csv=csv,
                n_per_arm=N_EXPECTED,
                families=FAMILIES,
                metrics={
                    m: dict(
                        column=COLUMN[m],
                        role=ROLE[m],
                        direction=DIRECTION[m],
                    )
                    for m in METRICS
                },
                diversity_descriptive_only=div_cols,
            ),
            f,
            indent=2,
        )

    print(f"[done] wrote -> {out}/")


if __name__ == "__main__":
    main()