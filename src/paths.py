"""Project-relative locations shared by data preparation and analysis modules."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

# The root is located from this module so the project can be moved as a directory.
ROOT = Path(__file__).resolve().parents[1]
REPO = str(ROOT)


def project_path(value):
    """Interpret a relative setting from the project root, not the caller's cwd."""
    return ROOT / Path(value)


DATA = str(project_path(os.environ.get('THESIS_DATA', 'data/raw')))
OUTPUT = str(project_path(os.environ.get('THESIS_OUTPUT', 'data/interim')))
PROCESSED = project_path('data/processed')
RESULTS = project_path('results')
API_KEY_FILE = str(project_path(os.environ.get('THESIS_API_KEY_FILE', 'API Key.txt')))
WRDS_USER_FILE = str(project_path(os.environ.get('THESIS_WRDS_USER_FILE', '.wrds_user')))
SEC_USER_AGENT_FILE = str(project_path(os.environ.get('THESIS_SEC_USER_AGENT_FILE', '.sec_user_agent')))
NO_API = os.environ.get('THESIS_NO_API', '1') != '0'


def data(*parts):
    return str(Path(DATA).joinpath(*parts))


def out(name, *parts):
    return str(Path(OUTPUT).joinpath(name, *parts))


def logs(name, *parts):
    return out(name, 'logs', *parts)


def code(*parts):
    return str(ROOT.joinpath(*parts))


def api_key():
    if NO_API:
        return ''
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if key:
        return key
    path = project_path(os.environ.get('THESIS_API_KEY_FILE', 'API Key.txt'))
    return path.read_text(encoding='utf-8').strip()


def wrds_user():
    """The WRDS username: WRDS_USER, the .wrds_user file, or the username field of the
    PostgreSQL password file that the wrds client reads the password from."""
    user = os.environ.get('WRDS_USER', '').strip()
    if user:
        return user
    path = project_path(os.environ.get('THESIS_WRDS_USER_FILE', '.wrds_user'))
    if path.is_file():
        return path.read_text(encoding='utf-8').strip()
    pgpass = (Path(os.environ['APPDATA']) / 'postgresql' / 'pgpass.conf' if os.name == 'nt'
              else Path.home() / '.pgpass')
    if pgpass.is_file():
        for line in pgpass.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split(':')
            if len(parts) >= 5 and 'wrds' in parts[0]:
                return parts[3]
    return None


def sec_user_agent():
    """The User-Agent of SEC EDGAR requests: SEC_USER_AGENT or the .sec_user_agent file.

    EDGAR asks every automated client to name itself and give a contact address, for
    example "University Name Research name@example.edu"."""
    agent = os.environ.get('SEC_USER_AGENT', '').strip()
    if not agent and Path(SEC_USER_AGENT_FILE).is_file():
        agent = Path(SEC_USER_AGENT_FILE).read_text(encoding='utf-8').strip()
    if not agent:
        raise SystemExit('SEC EDGAR requests need a User-Agent with a contact address: set '
                         'SEC_USER_AGENT or write it to %s' % SEC_USER_AGENT_FILE)
    return agent


def portable(path):
    """A path as recorded in logs and tables: relative to data/raw, data/interim or the
    project root, whatever THESIS_DATA and THESIS_OUTPUT point to, in forward slashes."""
    path = Path(path).resolve()
    for base, label in ((Path(OUTPUT).resolve(), 'data/interim'), (Path(DATA).resolve(), 'data/raw'),
                        (ROOT, '')):
        try:
            rel = path.relative_to(base).as_posix()
        except ValueError:
            continue
        return (label + '/' + rel) if label else rel
    return path.as_posix()


def no_api_stop(n, what):
    if NO_API and n:
        raise SystemExit(f'{n} {what} have no cached answer. Enable API calls to classify new items.')


@contextmanager
def gzip_text_writer(path):
    import gzip
    import io
    with open(path, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='\n') as stream:
                yield stream


def _sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _signature(files):
    paths = [Path(f).resolve() for f in files]
    if not paths:
        raise ValueError('A cache needs at least one source file')
    common = Path(os.path.commonpath([str(p.parent) for p in paths]))
    return sorted([[p.relative_to(common).as_posix(), _sha256(p)] for p in paths])


def cache_valid(cache, inputs):
    path = Path(str(cache) + '.inputs.json')
    if not Path(cache).is_file() or not path.is_file():
        return False
    try:
        expected = {'version': 2, 'inputs': _signature(inputs), 'cache_sha256': _sha256(cache)}
        return json.loads(path.read_text(encoding='utf-8')) == expected
    except (OSError, ValueError, TypeError):
        return False


def cache_written(cache, inputs):
    record = {'version': 2, 'inputs': _signature(inputs), 'cache_sha256': _sha256(cache)}
    Path(str(cache) + '.inputs.json').write_text(json.dumps(record, sort_keys=True) + '\n', encoding='utf-8')
