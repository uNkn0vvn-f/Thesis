"""Thematic funds, classified by a deterministic keyword taxonomy on the fund NAME.

ChatGPT reached every fund at once, so without a comparison group a time trend cannot be
separated from the event. Thematic funds provide one: ChatGPT is a shock to ONE theme. Clean energy, cannabis,
blockchain, space and cybersecurity funds are sold by the same sponsors, through the
same distribution, to the same retail clientele, and were not hit. So they are the
control group, and month fixed effects absorb everything common to the sector.

The theme comes from the fund's registered name by regular expression: deterministic,
checkable, and not exposed to the LLM classifier instability.
"""
import os, re

import numpy as np
import pandas as pd

from src import paths as tp  # noqa: E402

WK1 = tp.out('disclosure', 'fund_classifications_2020_2026.csv')
OUT = tp.out('holdings')
os.makedirs(OUT, exist_ok=True)

THEMES = [
    ('AI',            r'artificial\s+intelligence|\bA\.?I\.?\b|machine\s+learning|deep\s+learning|generative|\bLLM\b'),
    ('robotics',      r'\brobotic|\bautomation\b'),
    ('blockchain',    r'blockchain|crypto|bitcoin|ethereum|digital\s+asset|web\s?3'),
    ('clean energy',  r'clean\s+energy|solar|wind\s+energy|renewable|hydrogen|carbon|climate|green\s+energy'),
    ('EV',            r'electric\s+vehicle|\bEV\b|battery|lithium'),
    ('cannabis',      r'cannabis|marijuana|hemp|psychedelic'),
    ('space',         r'\bspace\b|aerospace\s+innovat|satellite'),
    ('cyber',         r'cyber\s?security|\bcyber\b'),
    ('genomics',      r'genomic|biotech\s+innovat|immunolog|precision\s+medicine'),
    ('metaverse',     r'metaverse|virtual\s+reality|augmented\s+reality|\bNFT\b|gaming|esports'),
    ('semis',         r'semiconductor|\bchip\b'),
    ('water',         r'\bwater\b'),
    ('infrastructure', r'infrastructure|\b5G\b|data\s+cent'),
    ('millennial/social', r'millennial|\bgen\s?z\b|social\s+media|sharing\s+economy'),
    ('disruptive',    r'disrupt|innovat|next\s+gen|thematic|megatrend|frontier\s+tech'),
]
BAD = re.compile(r'\bAIM\b|\bAIR\b|\bAID\b|\bSAI\b|\bMAI\b', re.I)
COMPILED = [(n, re.compile(p, re.I)) for n, p in THEMES]

# Every fund series with a dated prospectus in the classified corpus.
df = pd.read_csv(WK1, encoding='utf-8-sig', low_memory=False,
                 usecols=['series_id', 'series_name', 'filing_date', 'registrant_name',
                          'registrant_cik', 'ai_category'])
df['filing_date'] = pd.to_datetime(df.filing_date, errors='coerce')
df = df[df.filing_date.notna() & df.series_id.notna() & df.series_name.notna()]
df['nm'] = df.series_name.astype(str)


def tag(name):
    """The themes whose pattern matches a fund name, after removing words that only look
    like "AI"."""
    if BAD.search(name):
        name = BAD.sub(' ', name)
    hits = [n for n, rx in COMPILED if rx.search(name)]
    return hits


df['themes'] = df.nm.map(tag)
df['is_thematic'] = df.themes.map(len) > 0

# One row per fund: its first and last filing, registrant and themes; written to
# theme_universe.parquet.
f = (df.sort_values('filing_date').groupby('series_id')
       .agg(first=('filing_date', 'min'), last=('filing_date', 'max'), nm=('nm', 'first'),
            reg=('registrant_name', 'first'), cik=('registrant_cik', 'first'),
            themes=('themes', 'first')).reset_index())
f['is_thematic'] = f.themes.map(len) > 0
f['is_AI'] = f.themes.map(lambda t: 'AI' in t)
f['n_themes'] = f.themes.map(len)
f['yr'] = f['first'].dt.year
f['fm'] = f['first'].dt.to_period('M').dt.to_timestamp()
f['themes_str'] = f.themes.map(lambda t: '|'.join(t))
f.drop(columns=['themes']).to_parquet(os.path.join(OUT, 'theme_universe.parquet'), index=False)

# Report theme sizes and launch timing around ChatGPT.
print('series in corpus: %d' % len(f))
print('thematic by name: %d (%.1f%%)' % (f.is_thematic.sum(), 100 * f.is_thematic.mean()))
print('of which AI:      %d' % f.is_AI.sum())
print()
print('=' * 84)
print('THEME SIZES AND LAUNCH TIMING (first appearance in the corpus)')
print('=' * 84)
rows = []
for n, _ in THEMES:
    m = f.themes_str.str.contains(re.escape(n), regex=True)
    sub = f[m]
    pre = ((sub.fm >= '2021-01-01') & (sub.fm <= '2022-11-30')).sum()
    post = (sub.fm > '2022-11-30').sum()
    rows.append((n, len(sub), pre, post, post / max(pre, 1)))
r = pd.DataFrame(rows, columns=['theme', 'total', 'launch 2021-22', 'launch post', 'ratio'])
r = r.sort_values('total', ascending=False)
print('  %-20s %7s %14s %12s %8s' % ('theme', 'total', 'pre (23 mo)', 'post (40 mo)', 'post/pre'))
for _, x in r.iterrows():
    star = '   <-- treated' if x.theme == 'AI' else ''
    print('  %-20s %7d %14d %12d %8.2f%s'
          % (x.theme, x.total, x['launch 2021-22'], x['launch post'], x.ratio, star))

print()
print('=' * 84)
print('AI-NAMED LAUNCHES BY MONTH, vs ALL OTHER THEMES')
print('=' * 84)
th = f[f.is_thematic & (f.fm >= '2021-01-01')].copy()
th['grp'] = np.where(th.is_AI, 'AI', 'other theme')
m = th.groupby([th.fm.dt.to_period('Q').dt.to_timestamp(), 'grp']).size().unstack(fill_value=0)
m['AI share %'] = 100 * m.AI / (m.AI + m['other theme'])
print('  %-10s %6s %13s %12s' % ('quarter', 'AI', 'other themes', 'AI share %'))
for q in m.index:
    mark = '  <-- ChatGPT' if q.strftime('%Y-%m') == '2022-10' else ''
    print('  %-10s %6d %13d %12.1f%s'
          % (q.date(), m.loc[q, 'AI'], m.loc[q, 'other theme'], m.loc[q, 'AI share %'], mark))
