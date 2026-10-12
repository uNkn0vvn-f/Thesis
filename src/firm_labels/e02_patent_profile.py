"""What a firm actually patents, which is evidence rather than recall.

A firm's CPC profile identifies its business by COMPOSITION, and that is precisely what
a G06N-count measure cannot do. The three failures of a name-cued label all resolve on
sight:

    Walmart      G06Q(1127) commerce, G06F(402), B65G(70) conveying, A47F(43) shop
                 fittings. A retailer. The 36 G06N patents that a count measure reads as
                 AI are 3% of its portfolio.
    SoundHound   G10L(24) speech analysis, G06F(12). Exactly what the firm does, and
                 exactly what a G06N-only count scored as zero.
    Visa         G06Q(1424) payments, H04L(469) networks. 61 G06N patents, 3% of total.

This source reaches non-US firms that file no 10-K at all.

WINDOW. Patents filed from 2010, so a firm is described by what it has been working on
recently rather than by its whole history. Primary CPC subclass only - a patent is
counted once, under the class the examiner put first.

TITLES. The subclass codes are passed with plain-English titles for the 90 that appear
most often, which is 80% of all profile slots. Rarer codes are passed bare; the model
knows the CPC scheme well enough for those, and a missing title degrades gracefully into
the status quo rather than into an error.

The settings and titles here are used by p02_patent_pre, which builds the profile of each
information set from the patents filed up to its cutoff.
"""
import os

from src import paths as tp  # noqa: E402

UNI = os.environ.get('UNIVERSE') or tp.out('firm_labels', 'firm_universe.parquet')
PAT = tp.data('wrds', 'patents_permno_cpc.parquet')
SINCE = 2010
TOPN = 8

# Plain-English titles of the most frequent primary CPC subclasses.
CPC = {
    'G06F': 'digital data processing / computing',
    'H04L': 'digital data transmission, networking',
    'G06Q': 'data processing for commerce, business, payments',
    'H04W': 'wireless communication networks',
    'H04N': 'pictorial communication, video',
    'G06T': 'image data processing, computer graphics',
    'A61K': 'pharmaceutical preparations',
    'G06N': 'machine learning, neural networks, AI models',
    'G01N': 'analysing materials, analytical instruments',
    'A61B': 'diagnosis and surgery, medical devices',
    'G06V': 'image and video recognition, computer vision',
    'H04B': 'transmission systems',
    'C07K': 'peptides, proteins',
    'H01L': 'semiconductor devices',
    'G02B': 'optical elements and systems',
    'C12N': 'microorganisms, enzymes, genetic engineering',
    'C07D': 'heterocyclic chemistry',
    'A61M': 'devices for introducing media into the body',
    'B65D': 'containers and packaging',
    'G16H': 'healthcare informatics',
    'H04M': 'telephonic communication',
    'G01R': 'measuring electric and magnetic variables',
    'B01D': 'separation processes',
    'G01S': 'radar, sonar, lidar, radio direction-finding',
    'G10L': 'speech analysis and synthesis, audio recognition',
    'C07C': 'acyclic and carbocyclic chemistry',
    'C12Q': 'measuring involving enzymes or nucleic acids',
    'H01M': 'batteries, fuel cells',
    'H05K': 'printed circuits, electric apparatus casings',
    'A61F': 'implants, prostheses',
    'H10D': 'semiconductor devices',
    'B29C': 'shaping plastics',
    'E21B': 'earth and rock drilling, oil and gas wells',
    'H02J': 'power supply and distribution circuits',
    'H03K': 'pulse and logic circuits',
    'H02M': 'electric power conversion',
    'B01J': 'chemical processes, catalysts',
    'A61P': 'therapeutic activity of compounds',
    'G01C': 'navigation, measuring distances',
    'G11C': 'static stores, memory',
    'B60W': 'vehicle control systems',
    'B60L': 'electric propulsion of vehicles',
    'B60K': 'vehicle propulsion arrangements',
    'H01Q': 'antennas',
    'G05B': 'control and regulating systems',
    'B60R': 'vehicle fittings',
    'H01R': 'electric connectors',
    'G05D': 'controlling non-electric variables, autonomous control',
    'G08B': 'alarm and signalling systems',
    'G06K': 'data recognition, barcodes, RFID',
    'B62D': 'motor vehicles and trailers',
    'G01B': 'measuring length and angles',
    'C23C': 'coating metallic material',
    'B23K': 'soldering, welding, laser processing',
    'H01J': 'electric discharge tubes',
    'B32B': 'layered products',
    'A61L': 'sterilising and disinfecting',
    'C02F': 'water and waste-water treatment',
    'G07F': 'coin-operated and vending apparatus',
    'F16H': 'gearing and transmissions',
    'B25J': 'manipulators, industrial robots',
    'B65G': 'conveying, storing, warehouse handling',
    'A47F': 'shop fittings and display',
    'H04R': 'loudspeakers, microphones, transducers',
    'G09G': 'display control',
    'H04J': 'multiplex communication',
    'F24F': 'air conditioning and ventilation',
    'F28D': 'heat exchangers',
    'H02K': 'electric motors and generators',
    'H02S': 'photovoltaic power generation',
    'F03D': 'wind motors',
    'G21C': 'nuclear reactors',
    'A01N': 'agrochemicals, biocides',
    'C08L': 'polymer compositions',
    'C09D': 'coatings, inks',
    'B22F': 'metallic powder metallurgy, additive manufacturing',
    'G03F': 'photolithography',
    'H01S': 'lasers',
    'G02F': 'optical devices controlling light',
    'H03M': 'coding and decoding',
    'G11B': 'information storage, recording',
    'G07C': 'time or attendance registers',
    'A63F': 'games, video games',
    'G09B': 'educational and teaching apparatus',
    'E04B': 'building construction',
    'F16K': 'valves',
    'B64G': 'cosmonautics, spacecraft',
    'B64C': 'aeroplanes, aircraft',
    'F02C': 'gas turbines',
    'G01M': 'testing machines and structures',
    'A23L': 'foods and non-alcoholic beverages',
}
