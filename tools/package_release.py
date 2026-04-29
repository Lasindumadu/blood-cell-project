"""Assemble release artifacts into `release/` and create a zip.

Collects: README_project.md, tools, src, data (small selection), runs weights (if present), and reports.
"""
import shutil
from pathlib import Path
import zipfile


def collect_release(output_dir='release'):
    root = Path('.')
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    # Copy high-level files
    for name in ['README_project.md', 'requirements.txt']:
        if Path(name).exists():
            shutil.copy(name, out / name)

    # copy tools and src
    shutil.copytree('tools', out / 'tools')
    shutil.copytree('src', out / 'src')

    # include small sample of data (do not copy huge raw datasets)
    data_dir = Path('data')
    if data_dir.exists():
        (out / 'data').mkdir()
        # copy BCCD metadata and small sample
        for p in ['bccd.yaml']:
            srcp = data_dir / p
            if srcp.exists():
                shutil.copy(srcp, out / 'data' / p)

    # include runs weights if present (best.pt)
    runs = Path('runs')
    if runs.exists():
        # copy any best.pt files (small check)
        for p in runs.rglob('best.pt'):
            dest = out / p.relative_to('.')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)

    # Create zip
    zip_path = Path('release.zip')
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in out.rglob('*'):
            zf.write(f, f.relative_to(out))

    print(f'Release assembled into {out} and {zip_path}')


if __name__ == '__main__':
    collect_release()
