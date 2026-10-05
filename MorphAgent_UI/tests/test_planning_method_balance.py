"""Code + VLM runs must ask the planner for a mix; methods are not rebalanced later."""
import json
from pathlib import Path

import pytest

from main import _ensure_vlm_feature, _planning_method_instructions
from morphagent_ui.models import RunConfig
from nodes.prompt_gen import fill_template


def instructions(features_per_round, ratio=0.9):
    return _planning_method_instructions('both', features_per_round, ratio)


def test_single_method_runs_keep_their_restriction():
    assert "only choose the 'code' method" in _planning_method_instructions('code', 5, 0.9)
    assert "only choose the 'vlm' method" in _planning_method_instructions('vlm', 5, 0.9)


def test_the_default_ratio_is_one_vlm_in_ten(monkeypatch):
    monkeypatch.delenv('CODE_VLM_RATIO', raising=False)
    assert RunConfig().code_vlm_ratio == 0.9


@pytest.mark.parametrize('per_round,ratio,code,vlm', [
    (10, 0.9, 9, 1),
    (20, 0.9, 18, 2),
    (30, 0.9, 27, 3),
    (5, 0.9, 4, 1),    # fewer than ten still keeps one VLM feature
    (3, 0.9, 2, 1),
    (2, 0.9, 1, 1),
    (10, 0.5, 5, 5),
    (6, 1.0, 5, 1),    # clamped so VLM never disappears
    (6, 0.0, 1, 5),    # and code keeps a slot when there is room
])
def test_mixed_runs_request_a_split_that_keeps_both_methods(per_round, ratio, code, vlm):
    text = instructions(per_round, ratio)
    assert f'roughly {code} feature(s) with "method": "code"' in text
    assert f'{vlm} feature(s) with "method": "vlm"' in text
    assert 'each method must appear at least once' in text


def test_a_single_feature_round_asks_for_vlm():
    assert 'must use "method": "vlm"' in instructions(1)


def test_an_all_code_plan_gives_its_last_feature_to_vlm():
    plan = [{'name': f'f{i}', 'method': 'code'} for i in range(5)]
    features = _ensure_vlm_feature(plan)
    assert [f['method'] for f in features] == ['code'] * 4 + ['vlm']
    assert features[-1]['name'] == 'vlm_f4'
    assert plan[-1]['method'] == 'code'


def test_a_plan_that_already_has_vlm_is_left_alone():
    plan = [{'name': 'a', 'method': 'code'}, {'name': 'vlm_b', 'method': 'vlm'}]
    assert _ensure_vlm_feature(plan) is plan
    assert _ensure_vlm_feature([]) == []


def test_the_guidance_reaches_the_planning_prompt(monkeypatch):
    template = json.loads(
        (Path(__file__).resolve().parents[1] / 'knowledge/prompts/feature_planning.json').read_text()
    )['template']
    monkeypatch.setenv('MORPHAGENT_KNOWLEDGE_MODE', 'lite')
    prompt = fill_template(template, {
        'user_query': 'Measure tau aggregation',
        'available_methods': 'code, vlm',
        'method_instructions': instructions(10),
    })
    assert 'roughly 9 feature(s) with "method": "code"' in prompt
    assert '{method_instructions}' not in prompt
