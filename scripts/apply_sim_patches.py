"""Apply the version-pinned Lift/rendering patches without replacing repositories.

Run after git submodule update --init --recursive. --check is read-only.
Already-applied patches are accepted; conflicting local edits are never overwritten.
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'mimicgen': '72bd767c255545f462e7ccfb2731f2e5d4c1d9bb',
    'robomimic': 'e10526b9a40c78b41f1e37e60041dc0ec0a5f60f',
    'robosuite': 'b9d8d3de5e3dfd1724f4a0e6555246c460407daa',
}

def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)

def apply_patches(root=ROOT, check=False):
    root = Path(root)
    pending = []
    # Validate everything before writing anything.
    for name, pin in PINS.items():
        repo = root / name
        if not (repo / '.git').exists():
            raise RuntimeError(f'{repo}: initialize the pinned submodule first')
        head = git(repo, 'rev-parse', 'HEAD')
        if head.returncode or head.stdout.strip() != pin:
            raise RuntimeError(f'{name}: expected upstream commit {pin}')
        patch = root / 'patches' / f'{name}.patch'
        if not patch.exists():
            continue
        if git(repo, 'apply', '--reverse', '--check', str(patch)).returncode == 0:
            print(f'{name}: patch already applied')
            continue
        result = git(repo, 'apply', '--check', str(patch))
        if result.returncode:
            raise RuntimeError(f'{name}: patch conflicts with local files; nothing overwritten.\n{result.stderr}')
        pending.append((name, repo, patch))
    for name, repo, patch in pending:
        if not check:
            result = git(repo, 'apply', str(patch))
            if result.returncode:
                raise RuntimeError(result.stderr)
        print(f'{name}: {"ready to apply" if check else "applied"}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    apply_patches(check=parser.parse_args().check)
