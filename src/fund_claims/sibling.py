"""Does a sentence in one fund's evidence belong to a SIBLING fund of the same filing?

A trust's 485BPOS often closes with a shared section, "Additional information about the
Funds' principal investment strategies", that states every fund's 80% policy under a short
name: "the AI ETF has adopted a non-fundamental investment policy ...", "the Energy Income
ETF has adopted ...". When the upstream segmenter hands that section to one fund, the other
funds' AI sentences become that fund's claim. TrueShares Eagle Global Renewable Energy Income
ETF was labelled an AI developer and adopter this way.

THE RULE, narrow on purpose. Mentioning a sibling is not enough: a fund of funds lists the
sibling ETFs it holds, and that is its own strategy. A sentence is a sibling's only when the
sibling is what the sentence is ABOUT:

  * the sibling is the subject: "the <Name> ETF has adopted / invests / seeks / considers ...";
  * or the sentence introduces the sibling's own material: "the policy of the <Name> ETF",
    "Additional Information About the <Name> ETF", "(for the <Name> ETF)";

and the sentence does not name THIS fund anywhere. A reference belongs to a fund when every
distinguishing word of <Name> is a word of that fund's registered name, or the acronym of
consecutive words in it ("AI" for "Artificial Intelligence"). A reference matching no
registered sibling is left alone: the rule only removes text it can place elsewhere.
"""
import re

NAME = r"((?:[A-Z][\w\-\.’']*|&|and|of)(?:\s+(?:[A-Z][\w\-\.’']*|&|and|of)){0,8}?)"
KIND = r"(?:ETFs?|Funds?|Portfolios?)"
VERB = (r"(?:’s|'s)?\s+(?:has|have|had|will|shall|may|seeks?|invests?|generally|considers?|"
        r"defines?|is|tracks?|intends?|employs?|uses?|aims?|adopted|normally|relies|rely|"
        r"utilizes?|follows?|selects?|applies|apply)\b")
ABOUT_RES = [
    re.compile(r"\b[Tt]he\s+" + NAME + r"\s+" + KIND + VERB),
    re.compile(r"\b[Pp]olic(?:y|ies)\s+of\s+the\s+" + NAME + r"\s+" + KIND),
    re.compile(r"(?:[Ii]nformation\s+[Aa]bout|\(for)\s+the\s+" + NAME + r"\s+" + KIND),
]
DROP = {'the', 'and', 'of', '&', 'etf', 'etfs', 'fund', 'funds', 'portfolio', 'portfolios',
        'trust', 'series', 'shares', 'class', 'inc', 'llc'}


def words(name):
    return [w for w in re.findall(r"[a-z0-9]+", (name or '').lower().replace('&', ' and '))
            if w not in DROP]


def vocab(name):
    """The words of a registered fund name and the acronyms of runs of 2-4 of them."""
    ws = words(name)
    v = set(ws)
    for n in (2, 3, 4):
        for i in range(len(ws) - n + 1):
            v.add(''.join(w[0] for w in ws[i:i + n]))
    return v


def names_self(sentence, own_name):
    """The fund's own name, less its wrapper word, appears in the sentence."""
    core = re.sub(r"\s+(ETF|Fund|Portfolio)s?\s*$", '', (own_name or '').strip(), flags=re.I)
    return len(core) >= 6 and core.lower() in sentence.lower()


def about(sentence):
    """Distinguishing-word sets of the funds the sentence is about ("the Fund" gives none)."""
    out = []
    for rx in ABOUT_RES:
        for m in rx.finditer(sentence):
            r = set(words(m.group(1)))
            if r:
                out.append(r)
    return out


def sibling_sentence(sentence, own_name, sibling_names):
    """True if the sentence is about another series of the filing and not about this one."""
    if names_self(sentence, own_name):
        return False
    own = vocab(own_name)
    sibs = [vocab(s) for s in sibling_names if s and s != own_name]
    refs = about(sentence)
    if any(r <= own for r in refs):
        return False
    return any(any(r <= s for s in sibs) for r in refs)


HEADER_RES = [
    re.compile(r"[Ii]nformation\s+[Aa]bout\s+the\s+" + NAME + r"\s+" + KIND),
    re.compile(r"\(for\s+the\s+" + NAME + r"\s+" + KIND + r"\)"),
    re.compile(r"FUND\s+SUMMARY\s*[—–-]+\s*(?:the\s+)?" + NAME + r"\s+" + KIND, re.I),
]
def header_owner(sentence, own_name, sibling_names):
    """'self', 'sibling' or None: whose section a sentence opens, if it opens one.
    "Additional Information About the <Name> ETF", "<Index> (for the <Name> ETF)", "FUND
    SUMMARY - <Name> ETF" and a sentence that begins with a fund's full name open a section
    that runs until the next such header."""
    own = vocab(own_name)
    sibs = [vocab(s) for s in sibling_names if s and s != own_name]
    found = None
    # A sentence that merely BEGINS with a sibling's name is not treated as a header: tried,
    # it cut 42 of the Alger AI Enablers & Adopters Fund's own AI sentences, because a fund's
    # own section does not always begin with its registered name, so nothing closed the
    # sibling's. Only explicit headers open a section; the rest is checked by hand
    # (sibling_adjudications.csv).
    for rx in HEADER_RES:
        for m in rx.finditer(sentence):
            r = set(words(m.group(1)))
            if not r:
                continue
            if r <= own:
                found = 'self'                 # the last header in a sentence wins
            elif any(r <= s for s in sibs):
                found = 'sibling'
    return found


def own_sentences(sents, own_name, sibling_names):
    """Keep-mask over a fund's sentences: False for every sentence inside a sibling's section
    and for every sentence about a sibling. With no siblings registered, everything is kept."""
    if not sibling_names:
        return [True] * len(sents)
    keep, inside = [], False
    for s in sents:
        h = header_owner(s, own_name, sibling_names)
        if h is not None:
            inside = (h == 'sibling')
        keep.append(not inside and not sibling_sentence(s, own_name, sibling_names))
    return keep
