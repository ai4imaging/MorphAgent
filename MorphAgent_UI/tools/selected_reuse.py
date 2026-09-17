"""Execute only explicitly selected historical per-feature extractors.

No merged extractor, model planning, or VLM scoring is executed on this path.
The standalone feature functions may differ from a historical LLM-merged function;
the UI records that the original per-feature scripts are the reused source.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

from morphagent_ui.feature_outputs import selected_features
from tools.code_executor import CodeExecutor
from tools.code_reuse import (find_primary_image_paths, find_segmentation_paths,
                             list_sample_ids, resolve_dataset_root)


def run_selected_reuse(source_results, data_root, output_dir, feature_names, *, conda_env=None):
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
    executor = CodeExecutor(dataset, conda_env=conda_env)
    names = [f['name'] for f in chosen]
    completed = 0
    print(f'[Reuse] Selected {len(names)} feature(s) on {len(samples)} target sample(s).', flush=True)
    for item in chosen:
        name = item['name']
        round_number = item['round_number'] or 1
        round_dir = output/f'round_{round_number}'
        feature_dir = round_dir/'features'/name
        feature_dir.mkdir(parents=True, exist_ok=True)
        script = feature_dir/'extract.py'
        shutil.copy2(source/item['codePath'], script)
        plans.setdefault(round_number, []).append({
            'name':name, 'method':'code', 'description':item['description'], 'category':item['category']})
        for row in rows:
            sample = row['sample_id']
            print(f'[Reuse] {name} · {sample}', flush=True)
            image = Path(find_primary_image_paths(dataset/sample)[0])
            ok, value, error = executor.execute_single_sample(script, image, find_segmentation_paths(dataset/sample))
            try:
                value = float(value)
                ok = ok and math.isfinite(value)
            except (TypeError, ValueError):
                ok = False
            row[name] = value if ok else ''
            if not ok:
                errors.setdefault(name, {})[sample] = error or 'No finite scalar value returned.'
                print(f'[Reuse] [ERROR] {name} · {sample}: {errors[name][sample]}', flush=True)
            completed += 1
            print(f'[Compute] Progress {completed}/{len(names) * len(samples)}', flush=True)
        # These are new measurements, not a claim of validation on the new dataset.
        entries.append({'feature_id':item['feature_id'], 'name':name, 'actual_column_name':name,
                        'description':item['description'], 'category':item['category'], 'method':'code',
                        'latest_round':round_number, 'current_status':'retained' if name not in errors else 'dropped',
                        'live':name not in errors, 'source_paths':{'reused_from':str(source/item['codePath'])},
                        'decision_history':[{'reason_codes':['reused_without_revalidation'], 'validation_score':None}]})
        with (output/'features.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['sample_id', *names])
            writer.writeheader();writer.writerows(rows)
    for number, definitions in plans.items():
        (output/f'round_{number}/feature_plan.json').write_text(json.dumps({'features':definitions, 'reuse':True}, indent=2), encoding='utf-8')
    (output/'feature_registry.json').write_text(json.dumps({'entries':entries, 'reuse':True}, indent=2), encoding='utf-8')
    result = {'complete':not errors, 'selected_features':names, 'sample_ids':samples, 'errors':errors,
              'llm_calls':False, 'vlm_calls':False, 'source_results':str(source),
              'code_source':'individual_feature_extractors', 'validation':'not_revalidated'}
    (output/'reuse_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'[Reuse] {"[DONE]" if not errors else "[PARTIAL]"} Saved {len(names)} feature columns.', flush=True)
    return result
