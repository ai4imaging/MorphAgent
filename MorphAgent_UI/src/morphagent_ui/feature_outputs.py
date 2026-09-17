"""Read-only feature inventory and compact, portable export snapshots."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

from .models import FeatureCard, load_feature_cards


def local_file(root: Path, relative: Path) -> Path | None:
    root = root.resolve()
    path = root / relative
    if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
        return path
    return None


def feature_catalog(results: str | Path) -> list[dict]:
    root = Path(results).resolve()
    if root.is_file():
        if root.name != 'feature_value.csv' or root.is_symlink():
            return []
        with root.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'sample_id', 'feature_name', 'value', 'description', 'method'} <= set(reader.fieldnames or []):
                return []
            unique = {}
            for row in reader:
                name = row.get('feature_name', '').strip()
                if name and name not in unique:
                    unique[name] = {'feature_id':'value:' + name, 'name':name,
                                    'method':row.get('method', 'unknown'), 'category':'other',
                                    'description':row.get('description', ''), 'status':'retained',
                                    'round_number':0, 'validation_score':None,
                                    'codePath':None, 'reusable':False, 'code_status':'not_in_values'}
        return list(unique.values())
    definitions = local_file(root, Path('feature_descriptions.csv'))
    portable = bool(definitions)
    folders = {}
    cards = load_feature_cards(root)
    if portable:
        cards = []
        with definitions.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                name = row.get('name', '')
                if not name or name in folders or row.get('status') == 'dropped':
                    continue
                folders[name] = row.get('folder', '')
                try:
                    number = max(0, int(row.get('round_number', 0)))
                except (ValueError, TypeError):
                    number = 0
                try:
                    score = float(row.get('validation_score', ''))
                    if not math.isfinite(score):
                        score = None
                except (ValueError, TypeError):
                    score = None
                cards.append(FeatureCard(feature_id='export:' + name, name=name,
                    method=row.get('method', 'unknown'), category=row.get('category', 'other'),
                    description=row.get('description', ''), status=row.get('status', 'retained'),
                    round_number=number, validation_score=score))
    items = []
    for card in cards:
        if card.status == 'dropped':
            continue
        item = asdict(card)
        script = None
        # Never follow stale absolute source_paths from imported result manifests.
        name = card.name
        if name and '/' not in name and '\\' not in name and not name.startswith('.'):
            if portable:
                folder = folders[name]
                if folder and '/' not in folder and '\\' not in folder and not folder.startswith('.'):
                    script = local_file(root, Path(folder) / 'code' / 'extract.py')
            else:
                relative = Path(f'round_{card.round_number}') / 'features' / name / 'extract.py'
                script = local_file(root, relative)
        if script:
            try:
                tree = ast.parse(script.read_text(encoding='utf-8'))
                if not any(isinstance(n, ast.FunctionDef) and n.name == 'extract' for n in tree.body):
                    script = None
            except (OSError, UnicodeError, SyntaxError):
                script = None
        item['codePath'] = script.relative_to(root).as_posix() if script and card.method == 'code' else None
        item['reusable'] = bool(item['codePath'])
        item['code_status'] = 'available' if item['reusable'] else 'vlm_no_code' if card.method == 'vlm' else 'missing_code'
        items.append(item)
    return items


def selected_features(results: str | Path, names) -> list[dict]:
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError('Select at least one code feature.')
    available = {f['name']: f for f in feature_catalog(results)}
    result = []
    for name in dict.fromkeys(names):
        item = available.get(name)
        if not item or not item['reusable']:
            raise ValueError(f'No standalone reusable code for feature: {name}')
        result.append(item)
    return result


def feature_measurements(root: Path, card: dict) -> list[dict]:
    """Read a saved column without truncation or following external manifests."""
    root = root.resolve()
    if root.is_file() and root.name == 'feature_value.csv':
        with root.open(encoding='utf-8-sig', newline='') as stream:
            return [{'sample_id':row.get('sample_id', ''), 'value':row.get('value', '')}
                    for row in csv.DictReader(stream) if row.get('feature_name') == card['name']]
    candidates = [Path('features.csv'), Path('feature_values.csv'), Path('retained_features.csv')]
    definitions = local_file(root, Path('feature_descriptions.csv'))
    if definitions:
        with definitions.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('name') == card['name'] and row.get('folder'):
                    candidates.append(Path(row['folder']) / 'values.csv')
                    break
    candidates.append(Path(f"round_{card['round_number']}") / 'features.csv')
    rounds = [p for p in root.glob('round_*') if re.fullmatch(r'round_\d+', p.name)]
    candidates.extend(p.relative_to(root) / 'features.csv'
                      for p in sorted(rounds, key=lambda p: int(p.name[6:]), reverse=True))
    for relative in dict.fromkeys(candidates):
        path = local_file(root, relative)
        if not path:
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if card['name'] not in (reader.fieldnames or []):
                continue
            sample_column = 'sample_id' if 'sample_id' in reader.fieldnames else reader.fieldnames[0]
            return [{'sample_id': row.get(sample_column, ''), 'value': row.get(card['name'], '')}
                    for row in reader]
    return []


def feature_distribution(results: str | Path, name: str) -> dict:
    """Small histogram payload; individual sample measurements stay on disk."""
    root = Path(results).resolve()
    card = next((c for c in feature_catalog(root) if name in (c['name'], c['feature_id'])), None)
    if not card:
        raise ValueError('Unknown feature.')
    numbers, missing = [], 0
    for row in feature_measurements(root, card):
        try:
            value = float(row['value'])
            if not math.isfinite(value):
                raise ValueError('Non-finite measurement')
            numbers.append(value)
        except (TypeError, ValueError):
            missing += 1
    result = {'feature': {'name': card['name'], 'status': card['status']},
              'count': len(numbers), 'missing': missing, 'bins': [], 'minimum': None, 'maximum': None}
    if not numbers:
        return result
    low, high = min(numbers), max(numbers)
    result.update(minimum=low, maximum=high)
    size = min(30, max(1, math.ceil(math.sqrt(len(numbers)))))
    if low == high:
        size = 1
        padding = max(abs(low) * .05, .5)
        low, high = low - padding, high + padding
        # Keep JSON finite even at the limits of a float.
        if not math.isfinite(low): low = result['minimum']
        if not math.isfinite(high): high = result['maximum']
    scale = max(abs(low), abs(high), 1e-300)
    span = high / scale - low / scale
    edges = [low * (1 - i / size) + high * (i / size) for i in range(size + 1)]
    counts = [0] * size
    for value in numbers:
        index = int((value / scale - low / scale) / span * size)
        counts[max(0, min(size - 1, index))] += 1
    result['bins'] = [{'low': edges[i], 'high': edges[i + 1], 'count': count}
                      for i, count in enumerate(counts)]
    return result


def _folder_name(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_.-]', '_', name).strip(' ._')[:100] or 'feature'
    if cleaned != name or cleaned.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}:
        cleaned += '_' + hashlib.sha256(name.encode()).hexdigest()[:8]
    return cleaned


def _write_csv(path, fields, rows, redact):
    def clean(value):
        text = redact(str(value)) if value is not None else ''
        # Keep numeric negatives numeric; protect descriptions when opened in Excel.
        if text.startswith(('=', '+', '@', '\t', '\r')):
            text = "'" + text
        return text
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: clean(v) for k,v in row.items()} for row in rows)


def export_run(results, exports, timestamp, *, redact, run_status='unverified'):
    if not re.fullmatch(r'\d{8}_\d{6}(?:_\d{6})?', timestamp):
        raise ValueError('Invalid run timestamp.')
    root = Path(results).resolve()
    cards = feature_catalog(root)
    if not cards:
        raise ValueError('No feature results to save yet.')
    matrix = (root if root.is_file() and root.name == 'feature_value.csv' else None)
    matrix = (matrix or local_file(root, Path('features.csv')) or local_file(root, Path('retained_features.csv'))
              or local_file(root, Path('feature_values.csv')))
    if not matrix:
        raise ValueError('No feature values table was produced.')
    with matrix.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames or [], list(reader)
    if not fields:
        raise ValueError('The feature values table is empty.')
    exports = Path(exports)
    exports.mkdir(parents=True, exist_ok=True)
    destination = exports / timestamp
    counter = 2
    while destination.exists() or destination.with_suffix('.zip').exists():
        destination = exports / f'{timestamp}_{counter}'
        counter += 1
    destination.mkdir()
    feature_root = destination / 'feature'
    value_root = destination / 'value'
    feature_root.mkdir()
    value_root.mkdir()
    descriptions = []
    values = []
    used = set()
    for item in cards:
        name = item['name']
        folder = _folder_name(name)
        if folder.casefold() in used:
            folder += '_' + hashlib.sha256(item['feature_id'].encode()).hexdigest()[:8]
        used.add(folder.casefold())
        feature_dir = feature_root / folder
        code_dir = feature_dir / 'code'
        code_dir.mkdir(parents=True)
        values.extend({'sample_id':row['sample_id'], 'feature_name':name, 'value':row['value'],
                       'description':item['description'], 'method':item['method']}
                      for row in feature_measurements(root, item))
        code_status = item['code_status']
        if item['codePath']:
            (code_dir / 'extract.py').write_text(redact((root / item['codePath']).read_text(encoding='utf-8')), encoding='utf-8')
        else:
            # A VLM definition is not executable measurement code. Never fabricate it.
            (code_dir / 'definition.json').write_text(json.dumps({
                'feature':name, 'method':item['method'], 'description':redact(item['description']),
                'code_status':code_status,
                'note':'VLM scoring has no standalone measurement script.' if item['method']=='vlm' else 'No standalone extract.py was saved for this feature.'
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        descriptions.append({**item, 'folder':folder, 'code_status':code_status, 'run_status':run_status})
    _write_csv(feature_root / 'feature_descriptions.csv',
               ['name','description','method','category','status','round_number','validation_score','folder','code_status','run_status'], descriptions, redact)
    _write_csv(value_root / 'feature_value.csv',
               ['sample_id','feature_name','value','description','method'], values, redact)
    archive = destination.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(destination.rglob('*')):
            if file.is_file():
                bundle.write(file, (Path(destination.name) / file.relative_to(destination)).as_posix())
    return {'directory':str(destination), 'archive':str(archive), 'filename':archive.name,
            'featureCount':len(cards), 'codeCount':sum(f['reusable'] for f in cards)}
