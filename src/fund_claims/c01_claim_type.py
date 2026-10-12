"""What does an AI name actually promise? theme / process / overlay, per name spell.

The distinction the claim taxonomy rests on: a THEME name says what is HELD and holdings can
falsify it; a PROCESS name says WHO PICKS and holdings cannot. An OVERLAY - a leveraged,
inverse or option-income product written on an AI index - promises neither, and scoring
its equity sleeve against the Names Rule 80% bar is meaningless.

Two cases a plain keyword rule gets wrong, and this rule handles:

  * a PROCESS regex listing powered|enhanced|managed|driven|select|optimi[sz]ed|
    augmented|assisted|guided but not "Pilot" calls QRAFT AI-Pilot a theme fund;
  * without an OVERLAY category, fourteen leveraged, inverse and option-income products
    are scored as failed theme funds.

Classification is per NAME SPELL, not per fund, because a fund that renames can change
what it promises. Precedence is overlay > process > theme, with every borderline case
written out for adjudication rather than settled silently.
"""
import io, os, re
import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
LOGS = tp.logs('fund_claims')
UNI = tp.out('fund_claims')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# ---- OVERLAY: the payoff is engineered on top of something else ----------------
# Bare "bull"/"bear" are deliberately NOT here. Merlyn.AI Bull-Rider Bear-Fighter ETF
# is an algorithmic product whose name uses them as adjectives, and every genuinely
# leveraged bull/bear fund in this universe also carries an explicit multiplier, so
# the multiplier alone catches all of them and none of the false positives.
OVERLAY = re.compile(
    r'\b\d+(\.\d+)?\s?x\b|\binverse\b|\bultra(short|pro)\b|'
    r'\bdaily\s+target\b|\bleverage[d]?\s+shares\b|'
    r'\boption\s+income\b|\bpremium\s+income\b|\bcovered\s+call\b|\bbuffer\b|'
    r'\bdefined\s+outcome\b|\byieldmax\b|\bt-?rex\b',
    re.I)

# ---- PROCESS: the name says WHO PICKS ------------------------------------------
PROCESS = re.compile(
    r'\bAI[\s\-‐-―]?(powered|enhanced|managed|driven|select|selected|'
    r'optimi[sz]ed|augmented|assisted|guided|pilot|quant|sentiment)\b|'
    r'powered\s+by\s+AI|\bAI\s+smart\b|\bsmart\s+sentiment\b|'
    r'machine[\s-]?learning[\s-]?(powered|driven|enhanced|managed)|'
    # a brand that merely contains AI promises nothing about holdings either
    r'\w\.AI\b|\bMerlyn\b|\bEquBot\b|\bSectorSurfer\b|\bLet\s+Bob\b|\bStockSnips\b|'
    r'\bFINQ\b|\bAi\s+Funds\b',
    re.I)

# ---- cases a human should look at, whatever the rule decided -------------------
BORDERLINE = [
    (re.compile(r'\bAI\s+Quality\b', re.I),
     'is "AI" the theme or the manager?'),
    (re.compile(r'enablers|adopters', re.I),
     'theme, but "adopters" means users of AI, not producers'),
    (re.compile(r'\bDraco\s+Evolution\b|\bHarbor\s+AI\s+Inflection\b', re.I),
     'brand or theme, unclear'),
    (re.compile(r'\bAI\s+Pilot\b', re.I),
     'QRAFT algorithmic product: process, not theme'),
    (re.compile(r'\bpremium\s+income\b|\boption\s+income\b', re.I),
     'income overlay written on an AI index - holdings are not the promise'),
    (re.compile(r'\bphysical\s+AI\b|\bhumanoid\b|\bembodied\b', re.I),
     'robotics/embodied - AI theme or a different theme?'),
    (re.compile(r'\bAI\s+lab\s+ecosystem\b', re.I),
     'single-company ecosystem basket'),
]


def classify(name):
    """Kept only for continuity with the single-label counts."""
    n = str(name)
    if OVERLAY.search(n):
        return 'overlay'
    if PROCESS.search(n):
        return 'process'
    return 'theme'


# The single label above forces overlay > process > theme, and that precedence is
# false to the data: 5 of the 24 process-named funds also carry a stable AI-theme
# claim in their prospectus text, and 14 of the 15 overlays do. A leveraged product
# written on an AI index is BOTH an overlay AND a theme claim; forcing one discards
# the other. These four axes may all be true at once.
def axes(name):
    n = str(name)
    return {'ai_in_name': True,
            'name_suggests_process': bool(PROCESS.search(n)),
            'name_suggests_theme': not bool(PROCESS.search(n)),
            'is_overlay': bool(OVERLAY.search(n))}


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    S = pd.read_parquet(os.path.join(UNI, 'ai_name_spells.parquet'))
    U = pd.read_parquet(os.path.join(UNI, 'ai_universe.parquet'))
    A = S[S.ai & S.series_id.isin(U.index)].copy()
    A['claim'] = A.series_name.map(classify)
    ax = A.series_name.map(axes).apply(pd.Series)
    for c in ax.columns:
        A[c] = ax[c]

    w('=' * 96)
    w('CLAIM TYPE, PER AI NAME SPELL')
    w('=' * 96)
    w('  AI name spells : %d over %d series' % (len(A), A.series_id.nunique()))
    w()
    w('  %-10s %8s %8s' % ('claim', 'spells', 'series'))
    for c, v in A.groupby('claim'):
        w('  %-10s %8d %8d' % (c, len(v), v.series_id.nunique()))
    w()

    # a fund's claim can change across spells
    per = A.groupby('series_id').claim.nunique()
    w('  series whose claim type changes across name spells : %d' % int((per > 1).sum()))
    for sid in per[per > 1].index:
        w('    %s' % sid)
        for _, r in A[A.series_id == sid].sort_values('date_from').iterrows():
            w('       %-8s %-58s %s..%s' % (r.claim, str(r.series_name)[:58],
                                            r.date_from.date(), r.date_to.date()))
    w()

    # fund-level: the claim of its LAST AI spell, plus the set it ever made
    F = (A.sort_values('date_from').groupby('series_id')
          .agg(ai_in_name=('ai_in_name', 'last'),
               name_suggests_process=('name_suggests_process', 'last'),
               name_suggests_theme=('name_suggests_theme', 'last'),
               is_overlay=('is_overlay', 'last'),
               claim_last=('claim', 'last'),
               claim_first=('claim', 'first'),
               claims=('claim', lambda s: '|'.join(sorted(set(s)))),
               ai_name=('series_name', 'last'),
               registrant_name=('registrant_name', 'last'),
               first_ai_date=('date_from', 'min'),
               last_ai_date=('date_to', 'max')))

    w('-' * 96)
    w('FUND-LEVEL COUNTS')
    w('-' * 96)
    w('  %-34s %8s %8s %8s %8s' % ('', 'theme', 'process', 'overlay', 'total'))
    vc = F.claim_last.value_counts()
    w('  %-34s %8d %8d %8d %8d'
      % ('point-in-time, last claim', vc.get('theme', 0),
         vc.get('process', 0), vc.get('overlay', 0), len(F)))
    w()

    w('-' * 96)
    w('THE FOUR AXES, WHICH MAY OVERLAP')
    w('-' * 96)
    for c in ('ai_in_name', 'name_suggests_process', 'name_suggests_theme',
              'is_overlay'):
        w('  %-26s %4d funds' % (c, int(F[c].sum())))
    w('  overlay AND theme-suggesting name : %d'
      % int((F.is_overlay & F.name_suggests_theme).sum()))
    w('  overlay AND process-suggesting    : %d'
      % int((F.is_overlay & F.name_suggests_process).sum()))
    w()
    w('  The single `claim_last` label below forces overlay > process > theme and is')
    w('  kept only for continuity with the single-label counts. Use the axes.')
    w()

    w('-' * 96)
    w('EVERY FUND, BY CLAIM')
    w('-' * 96)
    for c in ('process', 'overlay', 'theme'):
        v = F[F.claim_last == c].sort_values('ai_name')
        w('  --- %s (%d) ---' % (c.upper(), len(v)))
        for sid, r in v.iterrows():
            w('     %s  %-64s' % (sid, str(r.ai_name)[:64]))
        w()

    w('-' * 96)
    w('BORDERLINE CASES FOR ADJUDICATION')
    w('-' * 96)
    rows = []
    for sid, r in F.iterrows():
        for rx, why in BORDERLINE:
            if rx.search(str(r.ai_name)):
                w('  [%-7s] %-58s' % (r.claim_last, str(r.ai_name)[:58]))
                w('             %s' % why)
                rows.append({'series_id': sid, 'ai_name': r.ai_name,
                             'rule_says': r.claim_last, 'reason': why,
                             'adjudicated': ''})
                break
    pd.DataFrame(rows).to_parquet(
        os.path.join(OUT, 'claim_borderline.parquet'), index=False)
    w()
    w('  %d borderline funds written to out/claim_borderline.parquet' % len(rows))
    w()

    A.to_parquet(os.path.join(OUT, 'claim_by_spell.parquet'), index=False)
    F.to_parquet(os.path.join(OUT, 'claim_by_fund.parquet'))
    w('wrote out/claim_by_spell.parquet (%d spells)' % len(A))
    w('wrote out/claim_by_fund.parquet  (%d funds)' % len(F))

    with io.open(os.path.join(LOGS, 'c01_claim_type.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()
