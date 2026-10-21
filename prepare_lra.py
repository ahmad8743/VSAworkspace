"""Download pinned original AAN TSVs from a mirror, then prepare Retrieval."""
import argparse
import shutil
import tarfile
import urllib.request
import time
from pathlib import Path

from lra_retrieval import prepare, digest

REVISION = '6cb133c44f406da743661488df9e977b87d98775'
BASE = f'https://huggingface.co/datasets/OpenNLPLab/lra/resolve/{REVISION}/data/aan/'
FILES = {
    'train': (8534323181, '0858d52d62320fa1aa07c2defb3124b86a5a6e1e5292a847f130356ea39d81d9'),
    'eval': (1046290902, '53f7008370276e26b7f1c2f74b187de76f1c6edd089d978ace7b946918fd259c'),
    'test': (1035013913, 'fef3c879f83d3e5173331dacbdd9a812a6d2af245396a1365dcb3e4b249831c9'),
}


def download(url, target, size, checksum):
    if target.exists():
        print(f'Checking existing {target.name} ...', flush=True)
        if target.stat().st_size == size and digest(target) == checksum:
            return
        raise ValueError(f'{target} has unexpected contents; move it aside before retrying')
    partial = target.with_suffix(target.suffix + '.partial')
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f'Partial file too large: {partial}')
        if offset == size:
            break
        headers = {'User-Agent': 'LRA-Retrieval-Downloader/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Server returned an incorrect resume range')
                    mode = 'ab'
                else:
                    # A server ignoring Range requires restarting, not appending.
                    offset, mode = 0, 'wb'
                last = time.monotonic()
                with partial.open(mode) as f:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                        offset += len(block)
                        if time.monotonic() - last >= 10:
                            print(f'{target.name}: {100 * offset / size:.1f}% ({offset / 1e9:.2f}/{size / 1e9:.2f} GB)', flush=True)
                            last = time.monotonic()
            if partial.stat().st_size == size:
                break
        except (OSError, urllib.error.URLError) as error:
            print(f'Download attempt {attempt + 1} interrupted: {error}', flush=True)
            if attempt == 2:
                raise
            time.sleep(2)
    if partial.stat().st_size != size or digest(partial) != checksum:
        raise ValueError(f'Incomplete or invalid download: {partial}; rerun to resume if incomplete')
    partial.replace(target)
    print(f'Verified {target.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--official-archive', action='store_true',
                        help='Use the original Google archive (currently returns HTTP 403)')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.official_archive:
        source = args.root / 'lra_tsv'
        source.mkdir(exist_ok=True)
        print('Downloading original AAN TSVs from pinned OpenNLPLab/lra mirror (10.6 GB total).', flush=True)
        for split, (size, checksum) in FILES.items():
            name = f'new_aan_pairs.{split}.tsv'
            download(BASE + name, source / name, size, checksum)
        prepare(source, args.root / 'lra_retrieval', 4000)
        return
    archive = args.root / 'lra_release.gz'
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        print('Downloading the official LRA release; this can take several minutes.', flush=True)
        with urllib.request.urlopen('https://storage.googleapis.com/long-range-arena/lra_release.gz') as response:
            with partial.open('wb') as target:
                shutil.copyfileobj(response, target)
        partial.replace(archive)
    source = args.root / 'lra_tsv'
    source.mkdir(exist_ok=True)
    expected = {f'new_aan_pairs.{s}.tsv' for s in ('train', 'eval', 'test')}
    found = set()
    # Copy only specific ordinary files; never extract arbitrary archive paths.
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in expected:
                if name in found:
                    raise ValueError(f'Duplicate split in archive: {name}')
                with tar.extractfile(member) as original, (source / name).open('wb') as target:
                    shutil.copyfileobj(original, target)
                found.add(name)
                print(f'Extracted {name}', flush=True)
    if found != expected:
        raise ValueError(f'Missing files in official release: {expected - found}')
    prepare(source, args.root / 'lra_retrieval', 4000)


if __name__ == '__main__':
    main()
