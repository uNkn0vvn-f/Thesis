"""Claim-holdings correspondence: sensitivity, selection and inference.

Pre-specified before estimation (recorded here, not chosen after seeing results):

  Sample      AI-theme fund-quarters (prospectus theme claim in force at the report date).
  Outcome     value-weighted share of the labelled sleeve in companies scoring >= 2 on
              the role; size-decile x sector adjusted share as a second outcome.
  Contrast    funds naming the role against other AI-theme funds in the same quarter.
  Statistic   (i) the figure's quantity: quarterly median(claimants) - median(others),
              averaged over quarters with at least five funds in each group;
              (ii) pooled OLS of exposure on the role indicator with quarter fixed effects.
  Inference   funds are resampled with their whole histories (fund-cluster bootstrap,
              1,999 draws, seed 20261011) for (i); fund-clustered standard errors for (ii).
              Quarter fixed effects / within-quarter differences remove common quarter
              movements. Holm adjustment across the three roles within each variant.
              The count of positive quarters is reported as description only.
  Variants    coverage thresholds 30/50/70/80/90%; labels with filing or patent evidence
              only; the common sample of the baseline and evidence-only definitions;
              excluding the Magnificent Seven; excluding each quarter's ten companies
              held by the most AI-theme funds; comparison against theme funds naming no
              role; size-and-sector adjusted outcome; one fund per near-duplicate
              portfolio pair; leave-one-registrant-out range.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.analysis.robustness.common import (ROLES, SEED, MAG7, TABLES, LABELS,  # noqa: E402
                                                 load_exposure_table, load_claim_panel,
                                                 load_holdings, load_labels, load_buckets,
                                                 build_exposure, attach_claims)

B = 1999
MIN_FUNDS = 5


def quarter_table(D, role, outcome, other_mask=None):
    """Quarterly medians as in the figure. other_mask restricts the comparison group."""
    rows = []
    for q, g in D.groupby('q', sort=True):
        c = g[g['chain_' + role]][outcome].dropna()
        o = g[~g['chain_' + role]]
        if other_mask is not None:
            o = o[other_mask.loc[o.index]]
        o = o[outcome].dropna()
        ok = len(c) >= MIN_FUNDS and len(o) >= MIN_FUNDS
        rows.append({'q': q, 'n_claim': len(c), 'n_other': len(o),
                     'med_claim': c.median() if ok else np.nan,
                     'med_other': o.median() if ok else np.nan})
    T = pd.DataFrame(rows)
    T['gap'] = T.med_claim - T.med_other
    return T


def _wmedian(values_sorted, W):
    """Median of each row's replicated sample; W is (B x n) integer weights."""
    cw = np.cumsum(W, axis=1)
    T = cw[:, -1]
    p1, p2 = (T - 1) // 2, T // 2
    i1 = (cw > p1[:, None]).argmax(axis=1)
    i2 = (cw > p2[:, None]).argmax(axis=1)
    m = 0.5 * (values_sorted[i1] + values_sorted[i2])
    m[T == 0] = np.nan
    return m


def bootstrap_gap(D, role, outcome, valid_q, other_mask=None, seed=SEED, draws=B):
    funds = np.array(sorted(D.series_id.unique()))
    fidx = {f: i for i, f in enumerate(funds)}
    rng = np.random.default_rng(seed)
    counts = np.zeros((draws, len(funds)), dtype=np.int64)
    for b in range(draws):
        counts[b] = np.bincount(rng.integers(len(funds), size=len(funds)),
                                minlength=len(funds))
    gaps = np.full((draws, len(valid_q)), np.nan)
    for j, q in enumerate(valid_q):
        g = D[D.q == q]
        c = g[g['chain_' + role]]
        o = g[~g['chain_' + role]]
        if other_mask is not None:
            o = o[other_mask.loc[o.index]]
        meds = []
        for grp in (c, o):
            grp = grp[grp[outcome].notna()].sort_values(outcome)
            vals = grp[outcome].to_numpy()
            W = counts[:, [fidx[f] for f in grp.series_id]]
            meds.append(_wmedian(vals, W))
        gaps[:, j] = meds[0] - meds[1]
    return np.nanmean(gaps, axis=1)


def ols_fe(D, role, outcome):
    d = D[[outcome, 'chain_' + role, 'q', 'series_id']].dropna().copy()
    d['x'] = d['chain_' + role].astype(float)
    d = d.rename(columns={outcome: 'y'})
    m = smf.ols('y ~ x + C(q)', data=d).fit(cov_type='cluster',
                                            cov_kwds={'groups': pd.factorize(d.series_id)[0]})
    return m.params['x'], m.bse['x'], m.pvalues['x'], len(d), d.series_id.nunique()


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def analyse(name, D, outcome='{role}', other=None, note=''):
    out, qrows = [], []
    for role in ROLES:
        col = outcome.format(role=role)
        mask = None if other is None else other(D, role)
        T = quarter_table(D, role, col, mask)
        valid = T.loc[T.gap.notna(), 'q'].tolist()
        G = T.gap.mean()
        rec = {'variant': name, 'role': role, 'outcome': col, 'note': note,
               'fund_quarters': len(D), 'funds': D.series_id.nunique(),
               'quarters_valid': len(valid), 'quarters_positive': int((T.gap > 0).sum()),
               'avg_median_gap': G}
        if valid:
            bs = bootstrap_gap(D, role, col, valid, mask)
            rec.update(boot_se=np.nanstd(bs, ddof=1),
                       ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5),
                       p_boot=min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))))
            last = T.loc[T.q == valid[-1]].iloc[0]
            rec.update(last_q=valid[-1], last_med_claim=last.med_claim,
                       last_med_other=last.med_other, last_n_claim=int(last.n_claim),
                       last_n_other=int(last.n_other))
        Dm = D if mask is None else D[D['chain_' + role] | mask]
        b, se, p, n, f = ols_fe(Dm, role, col)
        rec.update(ols_coef=b, ols_se=se, ols_p=p, ols_n=n, ols_funds=f)
        out.append(rec)
        qrows.append(T.assign(variant=name, role=role, outcome=col))
    R = pd.DataFrame(out)
    if 'p_boot' in R:
        R['p_boot_holm'] = holm(R.p_boot.fillna(1))
    R['ols_p_holm'] = holm(R.ols_p)
    return R, pd.concat(qrows, ignore_index=True)


def near_duplicates(H, D):
    """Pairs of sample funds whose same-quarter portfolios overlap by > 90% of value."""
    h = H.merge(D[['series_id', 'q']], on=['series_id', 'q'])
    w = h.groupby(['series_id', 'q', 'gvkey']).value.sum()
    w = w / w.groupby(level=[0, 1]).transform('sum')
    w = w.reset_index()
    pairs = []
    for q, g in w.groupby('q'):
        P = g.pivot_table(index='series_id', columns='gvkey', values='value', fill_value=0)
        A = P.to_numpy()
        ids = P.index.to_numpy()
        for i in range(len(ids)):
            ov = np.minimum(A[i][None, :], A[i + 1:]).sum(axis=1)
            for j in np.flatnonzero(ov > 0.9):
                pairs.append({'q': q, 'fund_a': ids[i], 'fund_b': ids[i + 1 + j],
                              'overlap': ov[j]})
    return pd.DataFrame(pairs)


def main():
    E = load_exposure_table()
    H, L, S = load_holdings(), load_labels(), load_buckets()
    log = {'labels': LABELS}

    base = attach_claims(build_exposure(H, L, buckets=S), E)
    ev = attach_claims(build_exposure(H, L, evidence_only=True, buckets=S), E)
    theme = base[base.theme_broad].copy()
    D50 = theme[theme.labelled >= 50].copy()
    log['baseline'] = {'funds': int(D50.series_id.nunique()), 'fund_quarters': len(D50),
                       'quarters': [D50.q.min(), D50.q.max(), int(D50.q.nunique())]}

    # ---------- evidence: record share versus held-value share ----------
    log['labels_without_evidence_company_years'] = int((~L.has_evidence).sum())
    log['labels_total_company_years'] = len(L)
    hs = H.merge(L[['gvkey', 'yr', 'has_evidence']], on=['gvkey', 'yr'], how='inner')
    hs = hs.merge(D50[['series_id', 'q']], on=['series_id', 'q'])
    log['no_evidence_value_share_in_sample_labelled_sleeve_pct'] = float(
        100 * hs.loc[~hs.has_evidence, 'value'].sum() / hs.value.sum())
    log['no_evidence_fundquarter_share_median_pct'] = float(D50.noevidence_share.median())
    log['no_evidence_fundquarter_share_p90_pct'] = float(D50.noevidence_share.quantile(.9))
    # the same in the full exposure table, for comparison
    hall = H.merge(L[['gvkey', 'yr', 'has_evidence']], on=['gvkey', 'yr'], how='inner')
    log['no_evidence_value_share_all_funds_labelled_sleeve_pct'] = float(
        100 * hall.loc[~hall.has_evidence, 'value'].sum() / hall.value.sum())
    # role labels among no-evidence company-years
    for r in ROLES:
        log['role_ge2_share_pct_' + r] = {
            'with_evidence': float(100 * (L.loc[L.has_evidence, r] >= 2).mean()),
            'without_evidence': float(100 * (L.loc[~L.has_evidence, r] >= 2).mean())}

    results, quarters = [], []

    def run(name, D, **kw):
        R, Q = analyse(name, D, **kw)
        results.append(R)
        quarters.append(Q)
        print(name, R[['role', 'avg_median_gap', 'ci_lo', 'ci_hi', 'p_boot', 'ols_coef',
                       'ols_se', 'quarters_valid', 'quarters_positive']].round(3)
              .to_string(index=False), flush=True)

    run('1 baseline (coverage >= 50%)', D50)
    for t in (30, 70, 80, 90):
        run('2 coverage >= %d%%' % t, theme[theme.labelled >= t].copy())
    evt = ev[ev.theme_broad]
    E50 = evt[evt.labelled >= 50].copy()
    run('3 evidence-only labels (coverage >= 50%)', E50)
    common = D50[['series_id', 'q']].merge(E50[['series_id', 'q']])
    run('4a common sample, baseline labels', D50.merge(common))
    run('4b common sample, evidence-only labels', E50.merge(common))
    run('5 size-and-sector adjusted outcome', D50, outcome='abn_{role}')
    run('5b size-and-sector adjusted, evidence-only', E50, outcome='abn_{role}')

    mag = pd.DataFrame({'gvkey': list(MAG7)})
    M = attach_claims(build_exposure(H, L, drop=mag, buckets=S), E)
    M = M[M.theme_broad & (M.labelled >= 50)].copy()
    run('6 excluding the Magnificent Seven', M)
    # each quarter's ten companies held by the most AI-theme sample funds
    hs2 = H.merge(D50[['series_id', 'q']], on=['series_id', 'q'])
    breadth = hs2.groupby(['q', 'gvkey']).series_id.nunique().rename('n').reset_index()
    top = breadth.sort_values(['q', 'n'], ascending=[True, False]).groupby('q').head(10)
    top[['q', 'gvkey', 'n']].to_csv(TABLES / 'top10_widely_held_by_quarter.csv', index=False)
    T10 = attach_claims(build_exposure(H, L, drop=top[['gvkey', 'q']], buckets=S), E)
    T10 = T10[T10.theme_broad & (T10.labelled >= 50)].copy()
    run('7 excluding each quarter\'s 10 most widely held', T10)

    def none_named(D, role):
        return ~D[['chain_' + r for r in ROLES]].any(axis=1)
    run('8 versus theme funds naming no role', D50, other=none_named)

    dup = near_duplicates(H, D50)
    dup.to_csv(TABLES / 'near_duplicate_pairs.csv', index=False)
    drop_ids = set()
    if len(dup):
        # keep the fund with more sample quarters in each pair
        nq = D50.groupby('series_id').size()
        for _, r in dup.drop_duplicates(['fund_a', 'fund_b']).iterrows():
            if r.fund_a in drop_ids or r.fund_b in drop_ids:
                continue
            drop_ids.add(r.fund_b if nq.get(r.fund_a, 0) >= nq.get(r.fund_b, 0) else r.fund_a)
    log['near_duplicate_pairs'] = int(len(dup.drop_duplicates(['fund_a', 'fund_b']))) if len(dup) else 0
    log['near_duplicate_funds_dropped'] = sorted(drop_ids)
    run('9 one fund per near-duplicate pair', D50[~D50.series_id.isin(drop_ids)].copy())

    # theme definition: AI as the fund's selection basis, and the boilerplate screen of claim_text_screen.py
    strict = E[['series_id', 'q', 'theme_strict']].copy()
    strict['theme_strict'] = strict.theme_strict.astype('boolean').fillna(False).astype(bool)
    DS = D50.merge(strict, on=['series_id', 'q'])
    run('10 strict theme (AI a selection basis)', DS[DS.theme_strict].copy())
    scr = TABLES / 'claim_filings_risk_screen.csv'
    if scr.exists():
        F = pd.read_csv(scr, dtype={'series_id': str})
        bad = set(zip(F.loc[F.flag_boilerplate, 'series_id'], pd.to_datetime(F.loc[F.flag_boilerplate, 'date'])))
        key = list(zip(D50.series_id, pd.to_datetime(D50.date)))
        flagged = np.array([k in bad for k in key])
        log['fund_quarters_matched_to_flagged_theme_filing'] = int(flagged.sum())
        log['funds_matched_to_flagged_theme_filing'] = int(D50.loc[flagged, 'series_id'].nunique())
        run('11 excluding boilerplate/risk-only theme matches', D50[~flagged].copy())

    R = pd.concat(results, ignore_index=True)
    Q = pd.concat(quarters, ignore_index=True)

    # ---------- leave-one-registrant-out (fund family proxy) ----------
    P = load_claim_panel(columns=['series_id', 'cik', 'date'])
    fam = (P.dropna(subset=['series_id']).sort_values('date')
           .groupby('series_id').cik.last().astype(str))
    D50['family'] = D50.series_id.map(fam).fillna('unknown')
    loo = []
    for role in ROLES:
        for f in sorted(D50.family.unique()):
            T = quarter_table(D50[D50.family != f], role, role)
            loo.append({'role': role, 'family_dropped': f, 'avg_median_gap': T.gap.mean(),
                        'quarters_valid': int(T.gap.notna().sum()),
                        'quarters_positive': int((T.gap > 0).sum()),
                        'funds_dropped': int(D50.loc[D50.family == f, 'series_id'].nunique())})
    LOO = pd.DataFrame(loo)
    LOO.to_csv(TABLES / 'correspondence_leave_one_registrant_out.csv', index=False)
    log['leave_one_family_out'] = {
        r: {'min': float(g.avg_median_gap.min()), 'max': float(g.avg_median_gap.max()),
            'family_at_min': g.loc[g.avg_median_gap.idxmin(), 'family_dropped'],
            'families': int(len(g))}
        for r, g in LOO.groupby('role')}

    # ---------- role overlap ----------
    ov = D50[['chain_' + r for r in ROLES]].astype(int)
    log['fund_claim_overlap_fund_quarters'] = {
        'developer_and_enabler': int((ov.chain_developer & ov.chain_enabler).sum()),
        'developer_and_adopter': int((ov.chain_developer & ov.chain_adopter).sum()),
        'enabler_and_adopter': int((ov.chain_enabler & ov.chain_adopter).sum()),
        'all_three': int(ov.all(axis=1).sum()), 'none': int((ov.sum(axis=1) == 0).sum()),
        'any': int((ov.sum(axis=1) > 0).sum())}
    lab = hs.merge(L[['gvkey', 'yr'] + list(ROLES)], on=['gvkey', 'yr'])
    tot = lab.value.sum()
    log['company_label_overlap_value_share_pct_in_sample'] = {
        'developer': float(100 * lab.loc[lab.developer >= 2, 'value'].sum() / tot),
        'enabler': float(100 * lab.loc[lab.enabler >= 2, 'value'].sum() / tot),
        'adopter': float(100 * lab.loc[lab.adopter >= 2, 'value'].sum() / tot),
        'developer_and_enabler': float(100 * lab.loc[(lab.developer >= 2) & (lab.enabler >= 2), 'value'].sum() / tot),
        'developer_and_adopter': float(100 * lab.loc[(lab.developer >= 2) & (lab.adopter >= 2), 'value'].sum() / tot),
        'enabler_and_adopter': float(100 * lab.loc[(lab.enabler >= 2) & (lab.adopter >= 2), 'value'].sum() / tot),
        'no_role': float(100 * lab.loc[(lab[list(ROLES)] < 2).all(axis=1), 'value'].sum() / tot)}
    # Magnificent Seven value share of the labelled sleeve, sample
    log['mag7_value_share_of_labelled_sleeve_pct'] = float(
        100 * lab.loc[lab.gvkey.isin(MAG7), 'value'].sum() / tot)
    log['mag7_role_labels_2025'] = L[(L.yr == 2025) & L.gvkey.isin(MAG7)][
        ['gvkey'] + list(ROLES) + ['has_evidence']].assign(
            name=lambda d: d.gvkey.map(MAG7)).to_dict('records')

    # ---------- selection ----------
    allE = E.copy()
    th_all = allE[allE.theme_broad]
    sel = {'theme_fund_quarters_with_holdings': int(len(th_all)),
           'theme_funds_with_holdings': int(th_all.series_id.nunique()),
           'theme_fund_quarters_below_50': int((th_all.labelled < 50).sum()),
           'theme_funds_never_reaching_50': int((th_all.groupby('series_id').labelled.max() < 50).sum()),
           'fund_quarters_claim_unknown_or_expired': int(allE.date.isna().sum()),
           'funds_with_any_unknown_claim_quarter': int(allE[allE.date.isna()].series_id.nunique())}
    Pc = load_claim_panel()
    Pc['date'] = pd.to_datetime(Pc.date, errors='coerce')
    Pc = Pc[Pc.series_resolved & Pc.series_id.notna()]
    tb = Pc[Pc.theme_broad.astype('boolean').fillna(False).astype(bool)
            & (Pc.date >= '2020-10-01') & (Pc.date <= '2026-03-31')]
    theme_ids = set(tb.series_id)
    held_ids = set(allE.series_id)
    hold_any = set(H.series_id)
    sel['theme_claim_funds_2020q4_2026q1'] = len(theme_ids)
    sel['of_which_with_any_holdings_in_exposure_table'] = len(theme_ids & held_ids)
    sel['of_which_in_extracted_equity_holdings'] = len(theme_ids & hold_any)
    sel['of_which_in_principal_sample'] = len(theme_ids & set(D50.series_id))
    roles_by_fund = tb.groupby('series_id')[['chain_developer', 'chain_enabler', 'chain_adopter']].any()
    ins = roles_by_fund.index.isin(D50.series_id)
    sel['role_naming_rate_pct_in_sample'] = (100 * roles_by_fund[ins].mean()).round(1).to_dict()
    sel['role_naming_rate_pct_outside_sample'] = (100 * roles_by_fund[~ins].mean()).round(1).to_dict()
    sel['funds_outside_sample'] = int((~ins).sum())
    # names of theme claimants without holdings in the exposure table
    nm = Pc.dropna(subset=['series_name']).groupby('series_id').series_name.last()
    miss = sorted(theme_ids - held_ids)
    first = Pc[Pc.theme_broad.astype('boolean').fillna(False).astype(bool)].groupby('series_id').date.min()
    W = pd.DataFrame({'series_id': miss, 'series_name': [nm.get(s, '') for s in miss]})
    # derivative-based structures by name: leveraged, inverse, single-stock, option income
    deriv = (r'\b\d(?:\.\d+)?x\b|bear|bull|short|inverse|leverage|option income|put ?write|'
             r'yield premium|yieldmax|income strategy|daily|weekly|monthly|quarterly|STKd|'
             r'managed futures|buffer|covered call|premium income|target etf|capped accelerated')
    W['derivative_like_by_name'] = W.series_name.fillna('').str.contains(deriv, case=False, regex=True)
    W['first_theme_claim'] = W.series_id.map(first)
    W.to_csv(TABLES / 'theme_claimants_without_holdings.csv', index=False)
    sel['without_holdings_derivative_like_by_name'] = int(W.derivative_like_by_name.sum())
    sel['without_holdings_other'] = int((~W.derivative_like_by_name).sum())
    sel['without_holdings_other_first_claim_2025_or_later'] = int(
        ((~W.derivative_like_by_name) & (W.first_theme_claim >= '2025-01-01')).sum())
    log['selection'] = sel

    R.to_csv(TABLES / 'correspondence_inference.csv', index=False, float_format='%.6g')
    Q.to_csv(TABLES / 'correspondence_quarterly_medians.csv', index=False, float_format='%.6g')
    (TABLES / 'correspondence_log.json').write_text(json.dumps(log, indent=2, default=str), encoding='utf-8')
    print(json.dumps(log, indent=2, default=str))


if __name__ == '__main__':
    main()
