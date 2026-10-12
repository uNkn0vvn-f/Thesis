"""Estimation helpers shared by the flow regressions and the event designs.

    ols_cluster   OLS with fund-clustered standard errors, degrees of freedom adjusted
                  for absorbed fixed effects
    make_dims     fixed-effect codes after dropping singleton cells
    absorb        partial out one or several fixed effects by alternating projections
    absorbed_k    absorbed parameters to charge to the degrees of freedom
    demean        group-mean removal used by `absorb`
    nw            time-series OLS with Newey-West standard errors (the alphas)

with two sample definitions used by every flow regression: MIN_TNA, the smallest
beginning-of-month net assets at which a monthly flow rate is used, and OVERLAY, the
fund-name pattern of leveraged, inverse and option-income products.
"""
import re

import numpy as np
import pandas as pd

MIN_TNA = 1e6          # $1m: below this a monthly flow rate is noise
NW_LAG = 3             # Newey-West lags of the monthly alphas

# Applied to every fund name, not only the AI-named ones: the largest fund at rung 5 is
# Direxion Daily Semiconductor Bull 3X Shares, which is not AI-named but whose
# volatility decay would dominate any value-weighted statistic for the group.
OVERLAY = re.compile(
    r'\b\d+(\.\d+)?\s?x\b|\binverse\b|\bultra(short|pro)\b|'
    r'\bdaily\s+target\b|\bleverage[d]?\s+shares\b|'
    r'\boption\s+income\b|\bpremium\s+income\b|\bcovered\s+call\b|\bbuffer\b|'
    r'\bdefined\s+outcome\b|\byieldmax\b|\bt-?rex\b', re.I)


def ols_cluster(y, X, cl, n_absorbed=0):
    """OLS with cluster-robust standard errors.

    `n_absorbed` counts fixed-effect parameters partialled out of y and X before they got
    here. They are not columns of X, so (N - K) would otherwise see only the visible
    regressors and understate the standard errors. Effects nested inside the clusters are
    estimated from within-cluster variation and must not be counted; `absorbed_k`
    applies that rule.
    """
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    df = pd.DataFrame(X * u[:, None])
    df['g'] = cl
    M = df.groupby('g', observed=True).sum().values
    G, N, K = M.shape[0], len(y), X.shape[1] + n_absorbed
    V = XtXi @ (M.T @ M) @ XtXi * (G / (G - 1)) * ((N - 1) / (N - K))
    return b, np.sqrt(np.clip(np.diag(V), 0, None)), V, G


def demean(M, codes, ngroups):
    """Subtract the group mean of every column of M (n x k), groups given by codes."""
    M = np.asarray(M, float)
    cnt = np.bincount(codes, minlength=ngroups).astype(float)
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        s = np.bincount(codes, weights=M[:, j], minlength=ngroups)
        out[:, j] = M[:, j] - (s / cnt)[codes]
    return out


def absorb(M, dims, tol=1e-9, maxit=200):
    """Sweep out several fixed effects by alternating projections.

    One interacted FE (sponsor x month) is a single dimension and converges in one
    pass. Two separate FEs (fund, month) do not - a fund-month cell holds exactly one
    observation, so they cannot be interacted, and they have to be swept alternately
    until the change dies out.
    """
    M = np.asarray(M, float).copy()
    if len(dims) == 1:
        return demean(M, dims[0][0], dims[0][1])
    for _ in range(maxit):
        prev = M.copy()
        for codes, ng in dims:
            M = demean(M, codes, ng)
        if np.max(np.abs(M - prev)) < tol:
            break
    return M


def absorbed_k(dims, cluster):
    """How many absorbed parameters to charge to the degrees of freedom.

    A fixed effect whose every level lies inside a single cluster is estimated from that
    cluster's own observations and adds no independent degrees of freedom, so it is
    excluded - the rule Wooldridge (2010) states and reghdfe's dof(cluster) implements.
    Fund effects clustered by fund are nested and cost nothing; sponsor x month effects
    clustered by fund are not, because one sponsor-month cell spans many funds.

    `dims` is what `make_dims` returned and `absorb` actually used, so this cannot drift
    out of step with the regression it is describing.
    """
    cl = pd.factorize(pd.Series(cluster).astype(str).values)[0]
    k = 0
    for codes, ng in dims:
        d = pd.DataFrame({'g': np.asarray(codes), 'c': cl})
        if d.groupby('g').c.nunique().max() > 1:
            k += ng
    return max(k - 1, 0) if k else 0


def make_dims(D, fes):
    """Codes for each FE dimension, after dropping observations that any dimension
    absorbs completely (a cell of one). Dropping cascades, so it iterates."""
    keep = np.ones(len(D), bool)
    while True:
        drop = np.zeros(len(D), bool)
        for keys in fes:
            g = D.loc[keep].groupby(keys, observed=True).ngroup().values
            cnt = np.bincount(g, minlength=g.max() + 1)
            bad = cnt[g] < 2
            if bad.any():
                idx = np.flatnonzero(keep)[bad]
                drop[idx] = True
        if not drop.any():
            break
        keep &= ~drop
        if keep.sum() < 100:
            break
    sub = D[keep]
    dims = []
    for keys in fes:
        c = pd.factorize(sub.groupby(keys, observed=True).ngroup().values)[0]
        dims.append((c, c.max() + 1))
    return sub, dims


def nw(y, X, lag=NW_LAG):
    """OLS with a constant and Newey-West standard errors."""
    X = np.column_stack([np.ones(len(y)), X])
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    S = (X * u[:, None]).T @ (X * u[:, None])
    for L in range(1, lag + 1):
        wgt = 1 - L / (lag + 1)
        A = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None])
        S += wgt * (A + A.T)
    V = XtXi @ S @ XtXi * len(y) / max(len(y) - X.shape[1], 1)
    return b, np.sqrt(np.clip(np.diag(V), 0, None))
