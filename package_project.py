"""Create a reviewable handoff ZIP with source, data and current verified results."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
target = ROOT / 'Wholesale_Customers_Project.zip'
files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.md','.txt','.ipynb'}]
files += [ROOT / '.gitignore', ROOT / 'data' / 'wholesale_customers.csv']
files += [p for folder in ['docs', 'tests'] for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
latest = ROOT / 'artifacts' / 'latest.txt'
if latest.exists():
    run = ROOT / 'artifacts' / latest.read_text(encoding='utf-8').strip()
    files += [latest] + [p for p in run.rglob('*') if p.is_file()]
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        archive.write(path, path.relative_to(ROOT))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
print(f'{target}: {len(set(files))} files, {target.stat().st_size} bytes')