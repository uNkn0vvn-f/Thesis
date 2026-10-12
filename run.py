"""Run an analysis, prepare source data, or export analytical tables."""
import argparse
import os
import sys

sys.dont_write_bytecode = True


def main(argv=None):
    from src import paths
    args = list(sys.argv[1:] if argv is None else argv)
    os.chdir(paths.ROOT)
    if args and args[0] == 'prepare':
        from src.preparation import main as prepare
        return prepare(args[1:])
    if args and args[0] == 'export':
        from src.data.export_tables import main as export
        return export(args[1:])
    if args and args[0] == 'analysis':
        args = args[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=['all', 'claims', 'holdings'], default='all')
    parser.add_argument('--labels', choices=['luna', 'deepseek'], default='luna',
                        help='model labels: luna (the thesis measurement) or deepseek (secondary)')
    parsed = parser.parse_args(args)
    from src.analysis import build
    build(paths.PROCESSED, paths.RESULTS, section=parsed.section, labels=parsed.labels)
    sub = '' if parsed.labels == 'luna' else parsed.labels + '/'
    print('Tables: results/tables/%s\nFigures: results/figures/%s' % (sub, sub))


if __name__ == '__main__':
    main()
