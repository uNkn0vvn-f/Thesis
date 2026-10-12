"""Does the sign reversal survive when the AI labels are reassigned across firms?

shift_share finds that neither shock, on its own, stands out from what the same AI values would
produce if they were handed to other firms: holding the portfolios fixed, permuting the
labels across firms gives slopes as large as the real one about half the time. That is the
shift-level question Borusyak, Hull and Jaravel say a shift-share design has to answer, and
for a single shock the answer is no.

The paper's causal claim was never a single shock. It is the reversal: AI exposure attracts
flows after ChatGPT and loses them after DeepSeek, so a standing preference for whatever
the exposed funds happen to hold - which moves flows the same way in both windows - cancels
out of the difference. This tests exactly that. ONE permutation of firm identities is
applied to both shocks' labels at once, so a firm that is given NVIDIA's label is given it
in both windows, and for every permutation

    a = (b_ChatGPT + b_DeepSeek) / 2     the standing premium
    b = (b_ChatGPT - b_DeepSeek) / 2     the response to the shock's direction

is recomputed from the two cross-sections of shift_share. The real b is then ranked against the
permutation distribution. If the single-shock noise is the same stocks moving flows the
same way twice, the two permuted slopes move together and b's null distribution is narrow;
if not, b fails too.

Only firms labelled in both vintages are permuted; the rest keep their labels, so every
fund's covered share is unchanged. PERM=techsize permutes within technology x size
quintile (size on 29 November 2022, before both shocks), PERM=tech within technology /
other.
"""
import io
import os
import sys

import numpy as np
import pandas as pd

from src.analysis.events import shift_share as S1                                   # noqa: E402

BAL = os.environ.get('BAL', '1') == '1'
N_PERM = int(os.environ.get('N_PERM', '2000'))
PERM = os.environ.get('PERM', 'techsize')


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    d1, d2 = S1.build('chatgpt', BAL), S1.build('deepseek', BAL)
    common = sorted(set(d1['firms']) & set(d2['firms']))
    p1 = {k: j for j, k in enumerate(d1['firms'])}
    p2 = {k: j for j, k in enumerate(d2['firms'])}
    i1 = np.array([p1[k] for k in common])
    i2 = np.array([p2[k] for k in common])
    tech = d2['tech_k'][i2]

    if PERM == 'techsize':
        # Size is read on 29 November 2022, the last trading day before ChatGPT: dated before
        # both windows' shocks, so the strata carry no information from either.
        D, CL = S1.crsp_day(None)
        ev = pd.Timestamp('2022-11-29')
        dd = D[D.dlycaldt == ev][['permno', 'dlycap']].dropna()
        assert len(dd), 'no CRSP caps on the size date'
        dd = dd.merge(CL[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='permno')
        dd = dd[(ev >= dd.linkdt) & (ev <= dd.linkenddt)]
        cap = dd.groupby('gvkey').dlycap.max().reindex(common)
        qn = pd.qcut(cap.rank(method='first'), 5, labels=False).fillna(-1).to_numpy()
        key = tech * 10 + qn
    else:
        key = tech
    strata = [np.flatnonzero(key == v) for v in np.unique(key)]

    def slopes(g1, g2):
        b1, s1 = S1.unit_ols(d1['y'], d1['S'] @ g1, d1['Wu'])
        b2, s2 = S1.unit_ols(d2['y'], d2['S'] @ g2, d2['Wu'])
        return b1, b2, s1, s2

    b1, b2, s1, s2 = slopes(d1['g_lab'], d2['g_lab'])
    a_obs, b_obs = (b1 + b2) / 2, (b1 - b2) / 2
    # Studentised versions. A permuted label vector usually has less variance than the real
    # one (NVIDIA's 72.5% lands on a less-held stock), which rescales the raw slope and
    # widens its null distribution for reasons unrelated to the question; a t-statistic is
    # close to pivotal and does not have that problem.
    tb_obs = (b1 - b2) / np.sqrt(s1 ** 2 + s2 ** 2)
    t1_obs, t2_obs = b1 / s1, b2 / s2

    rng = np.random.default_rng(20260924)
    P = np.empty((N_PERM, 2))
    T = np.empty((N_PERM, 3))
    for r in range(N_PERM):
        perm = np.arange(len(common))
        for ix in strata:
            perm[ix] = ix[rng.permutation(len(ix))]
        g1, g2 = d1['g_lab'].copy(), d2['g_lab'].copy()
        g1[i1] = d1['g_lab'][i1[perm]]
        g2[i2] = d2['g_lab'][i2[perm]]
        q1, q2, e1, e2 = slopes(g1, g2)
        P[r] = (q1, q2)
        T[r] = (q1 / e1, q2 / e2, (q1 - q2) / np.sqrt(e1 ** 2 + e2 ** 2))
    A, Bd = (P[:, 0] + P[:, 1]) / 2, (P[:, 0] - P[:, 1]) / 2

    def pval(obs, dist):
        return (1 + np.sum(np.abs(dist) >= abs(obs))) / (1 + len(dist))

    w('=' * 96)
    w('THE SIGN REVERSAL UNDER ONE PERMUTATION OF FIRM LABELS ACROSS BOTH SHOCKS%s'
      % ('   |   BALANCED' if BAL else ''))
    w('=' * 96)
    w('  ChatGPT  : %d funds, %d labelled firms' % (len(d1['funds']), len(d1['firms'])))
    w('  DeepSeek : %d funds, %d labelled firms' % (len(d2['funds']), len(d2['firms'])))
    w('  firms labelled in both, permuted jointly: %d, in %d strata (%s)'
      % (len(common), len(strata),
         'technology x size quintile' if PERM == 'techsize' else 'technology / other'))
    w('  permutations: %d' % N_PERM)
    w()
    w('  %-34s %9s %12s %9s' % ('', 'observed', 'perm. sd', 'p'))
    w('  %-34s %+9.4f %12.4f %9.4f' % ('slope after ChatGPT', b1, P[:, 0].std(), pval(b1, P[:, 0])))
    w('  %-34s %+9.4f %12.4f %9.4f' % ('slope after DeepSeek', b2, P[:, 1].std(), pval(b2, P[:, 1])))
    w('  %-34s %+9.4f %12.4f %9.4f' % ('a, standing premium', a_obs, A.std(), pval(a_obs, A)))
    w('  %-34s %+9.4f %12.4f %9.4f' % ('b, response to direction', b_obs, Bd.std(), pval(b_obs, Bd)))
    w()
    w('  STUDENTISED (the statistic to read; see the comment in the code)')
    w('  %-34s %+9.2f %12s %9.4f' % ('t, slope after ChatGPT', t1_obs, '', pval(t1_obs, T[:, 0])))
    w('  %-34s %+9.2f %12s %9.4f' % ('t, slope after DeepSeek', t2_obs, '', pval(t2_obs, T[:, 1])))
    w('  %-34s %+9.2f %12s %9.4f' % ('t, response to direction b', tb_obs, '', pval(tb_obs, T[:, 2])))
    w('  (for reference: the same t-statistics against a normal: p = %.4f, %.4f, %.4f)'
      % tuple(2 * (1 - __import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(v)))
              for v in (t1_obs, t2_obs, tb_obs)))
    w()
    w('  correlation of the two permuted slopes across permutations: %+.3f'
      % np.corrcoef(P[:, 0], P[:, 1])[0, 1])
    w('  (positive: the same reassigned labels move flows the same way in both windows, which')
    w('   is the standing-preference noise that the difference b is built to cancel)')

    R = pd.DataFrame({'stat': ['slope ChatGPT', 'slope DeepSeek', 'a', 'b',
                               't ChatGPT', 't DeepSeek', 't b'],
                      'observed': [b1, b2, a_obs, b_obs, t1_obs, t2_obs, tb_obs],
                      'perm_sd': [P[:, 0].std(), P[:, 1].std(), A.std(), Bd.std(),
                                  T[:, 0].std(), T[:, 1].std(), T[:, 2].std()],
                      'p': [pval(b1, P[:, 0]), pval(b2, P[:, 1]), pval(a_obs, A),
                            pval(b_obs, Bd), pval(t1_obs, T[:, 0]), pval(t2_obs, T[:, 1]),
                            pval(tb_obs, T[:, 2])]})
    sfx = ('balanced' if BAL else 'all') + '_' + PERM
    R.to_parquet(os.path.join(S1.OUT, 'two_shock_permutation_%s.parquet' % sfx), index=False)
    w()
    w('wrote out/two_shock_permutation_%s.parquet' % sfx)
    with io.open(os.path.join(S1.LOGS, 'shift_share_two_events_%s.txt' % sfx), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()
