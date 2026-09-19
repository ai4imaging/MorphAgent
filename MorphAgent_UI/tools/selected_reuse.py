"""Execute only explicitly selected historical feature extractors.

Code features replay their saved standalone script, dispatched from one mechanically
merged module so a sample costs a single sandbox. VLM features have no script, so
they are rescored against the new images with the same batched call the original run
used; the saved description is the only thing carried over. No planning or validation
runs on this path, and the standalone code functions may differ from a historical
LLM-merged function, so the UI records the per-feature scripts as the reused source.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from morphagent_ui.feature_outputs import selected_features
from tools.code_executor import CodeExecutor
from tools.code_reuse import (find_primary_image_paths, find_segmentation_paths,
                             list_sample_ids, resolve_dataset_root)

VLM_ATTEMPTS = 3


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_values(output, names, rows):
    with (output/'features.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', *names])
        writer.writeheader();writer.writerows(rows)


_MERGED_TEMPLATE = '''"""Mechanically merged from the saved per-feature extractors.

Compute may not call an LLM, so rather than asking one to fuse the round the way
a discovery run does, every saved function is loaded exactly as it was written
and dispatched in turn. One sandbox per sample then replaces one per feature and
sample, and the image is decoded once instead of once per feature.
"""
_SOURCES = __SOURCES__
_EXTRACTORS = []
_UNAVAILABLE = {}

for _name, _path in _SOURCES:
    _namespace = {}
    try:
        with open(_path, encoding="utf-8") as _handle:
            exec(compile(_handle.read(), _path, "exec"), _namespace)
        _EXTRACTORS.append((_name, _namespace["extract"]))
    except Exception as _exc:
        _UNAVAILABLE[_name] = "{}: {}".format(type(_exc).__name__, _exc)


def extract_all(img, seg):
    results = dict(_UNAVAILABLE)
    for _name, _extract in _EXTRACTORS:
        try:
            results[_name] = _extract(img, seg)
        except Exception as _exc:
            results[_name] = "{}: {}".format(type(_exc).__name__, _exc)
    return results
'''


def _merged_extractor_source(names, scripts):
    """Build one `extract_all(img, seg)` that calls every saved extractor."""
    sources = [(name, str(Path(script).resolve())) for name, script in zip(names, scripts)]
    return _MERGED_TEMPLATE.replace('__SOURCES__', repr(sources))


def _registry_entry(item, method, status, source):
    # These are new measurements, not a claim of validation on the new dataset.
    return {'feature_id':item['feature_id'], 'name':item['name'], 'actual_column_name':item['name'],
            'description':item['description'], 'category':item['category'], 'method':method,
            'latest_round':item['round_number'] or 1, 'current_status':status, 'live':status == 'retained',
            'source_paths':{'reused_from':source},
            'decision_history':[{'reason_codes':['reused_without_revalidation'], 'validation_score':None}]}


def _score_vlm_features(dataset, samples, items, rows, errors, output, question, concurrency, progress):
    """Batch every selected VLM feature per sample, mirroring the discovery run."""
    from config import apply_vlm_provider
    from nodes.execution import _execute_vlm_features_batch
    from tools.segmentation import check_segmentation_exists
    from utils_helpers import select_appropriate_data_source

    apply_vlm_provider('online')
    log_dir = output/'vlm_batch'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir/'execution_log.txt'
    definitions = [{'name':i['name'], 'description':i['description'], 'category':i['category'],
                    'method':'vlm'} for i in items]
    by_sample = {row['sample_id']: row for row in rows}

    def score(sample):
        sample_dir = dataset/sample
        images = select_appropriate_data_source(sample_dir, definitions[0], '')
        if not images:
            return sample, {}, 'No image files usable for VLM scoring.'
        mask = check_segmentation_exists(sample_dir)
        state = {'messages':[], 'user_query':question, 'sample_id':sample, 'image_paths':images,
                 'research_summary':'', 'expert_examples':[], 'expert_knowledge':None,
                 'deep_research':None, 'rag_knowledge':None, 'feature_plan':{'features':definitions},
                 'segmentation_mask':str(mask) if mask else None, 'analysis_results':{},
                 'current_step':'execution', 'iteration_count':0, 'error_log':[],
                 'features_list':definitions, 'num_features':len(definitions)}
        for attempt in range(1, VLM_ATTEMPTS + 1):
            try:
                scores = _execute_vlm_features_batch(definitions, images, state,
                                                     segmentation_mask=str(mask) if mask else None,
                                                     log_file=str(log_file))
                return sample, scores or {}, None if scores else 'Empty VLM response.'
            except Exception as exc:  # noqa: BLE001 — one sample must not abort the run
                if attempt >= VLM_ATTEMPTS:
                    return sample, {}, f'{type(exc).__name__}: {exc}'
                time.sleep(1)
        return sample, {}, 'VLM scoring did not return a result.'

    print(f'[Reuse] Scoring {len(items)} VLM feature(s) with {concurrency} concurrent sample(s).', flush=True)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for future in as_completed({pool.submit(score, s): s for s in samples}):
            sample, scores, failure = future.result()
            row = by_sample[sample]
            for item in items:
                name = item['name']
                value = _finite(scores.get(name))
                row[name] = value if value is not None else ''
                if value is None:
                    reason = failure or 'No finite score returned.'
                    errors.setdefault(name, {})[sample] = reason
                    print(f'[Reuse] [ERROR] {name} · {sample}: {reason}', flush=True)
            progress(len(items))


def run_selected_reuse(source_results, data_root, output_dir, feature_names, *, conda_env=None,
                       question='', vlm_concurrency=1):
    source = Path(source_results).resolve()
    chosen = selected_features(source, feature_names)
    dataset = resolve_dataset_root(data_root)
    samples = list_sample_ids(data_root)
    if not samples:
        raise ValueError('Target dataset has no samples.')
    missing = [s for s in samples if not find_primary_image_paths(dataset/s)]
    if missing:
        raise ValueError('Target samples without primary images: '+', '.join(missing))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [{'sample_id':s} for s in samples]
    errors, plans, entries = {}, {}, []
    names = [f['name'] for f in chosen]
    code_items = [f for f in chosen if f['method'] != 'vlm']
    vlm_items = [f for f in chosen if f['method'] == 'vlm']
    total = len(names) * len(samples)
    counter = {'done':0}
    counter_lock = threading.Lock()

    def advance(step=1):
        with counter_lock:
            counter['done'] += step
            print(f'[Compute] Progress {counter["done"]}/{total}', flush=True)

    print(f'[Reuse] Selected {len(names)} feature(s) on {len(samples)} target sample(s).', flush=True)
    executor = CodeExecutor(dataset, conda_env=conda_env)
    if code_items:
        code_names, scripts = [], []
        for item in code_items:
            name = item['name']
            round_number = item['round_number'] or 1
            feature_dir = output/f'round_{round_number}'/'features'/name
            feature_dir.mkdir(parents=True, exist_ok=True)
            script = feature_dir/'extract.py'
            shutil.copy2(source/item['codePath'], script)
            code_names.append(name)
            scripts.append(script)
            plans.setdefault(round_number, []).append({
                'name':name, 'method':'code', 'description':item['description'],
                'category':item['category']})
        merged_dir = output/'merged_features'
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged = merged_dir/'extract_all.py'
        merged.write_text(_merged_extractor_source(code_names, scripts), encoding='utf-8')
        print(f'[Reuse] Replaying {len(code_names)} code feature(s), one sandbox per sample.', flush=True)
        for index, row in enumerate(rows, 1):
            sample = row['sample_id']
            image = Path(find_primary_image_paths(dataset/sample)[0])
            ok, values, error = executor.execute_single_sample(
                merged, image, find_segmentation_paths(dataset/sample))
            if not ok or not isinstance(values, dict):
                # The sandbox itself failed, so every feature of this sample is lost.
                error = error or 'Merged extractor did not return a feature mapping.'
                values = {}
            for name in code_names:
                raw = values.get(name)
                number = _finite(raw)
                row[name] = number if number is not None else ''
                if number is None:
                    reason = raw if isinstance(raw, str) and raw else (
                        error or 'No finite scalar value returned.')
                    errors.setdefault(name, {})[sample] = reason
                    print(f'[Reuse] [ERROR] {name} · {sample}: {reason}', flush=True)
            advance(len(code_names))
            if index % 25 == 0:
                _write_values(output, names, rows)
        _write_values(output, names, rows)

    if vlm_items:
        for item in vlm_items:
            plans.setdefault(item['round_number'] or 1, []).append({
                'name':item['name'], 'method':'vlm', 'description':item['description'],
                'category':item['category']})
        _score_vlm_features(dataset, samples, vlm_items, rows, errors, output, question,
                            vlm_concurrency, advance)
        _write_values(output, names, rows)

    for item in chosen:
        method = 'vlm' if item['method'] == 'vlm' else 'code'
        status = 'retained' if item['name'] not in errors else 'dropped'
        origin = str(source/item['codePath']) if method == 'code' else str(source)
        entries.append(_registry_entry(item, method, status, origin))
    for number, definitions in plans.items():
        (output/f'round_{number}'/'feature_plan.json').parent.mkdir(parents=True, exist_ok=True)
        (output/f'round_{number}'/'feature_plan.json').write_text(
            json.dumps({'features':definitions, 'reuse':True}, indent=2), encoding='utf-8')
    (output/'feature_registry.json').write_text(json.dumps({'entries':entries, 'reuse':True}, indent=2), encoding='utf-8')
    result = {'complete':not errors, 'selected_features':names, 'sample_ids':samples, 'errors':errors,
              'llm_calls':False, 'vlm_calls':bool(vlm_items), 'source_results':str(source),
              'code_source':'individual_feature_extractors', 'validation':'not_revalidated'}
    (output/'reuse_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'[Reuse] {"[DONE]" if not errors else "[PARTIAL]"} Saved {len(names)} feature columns.', flush=True)
    return result
