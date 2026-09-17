"""Code + VLM runs must ask the planner for a mix; methods are not rebalanced later."""
import json
from pathlib import Path

import pytest

from main import _planning_method_instructions
from nodes.prompt_gen import fill_template


def instructions(features_per_round, ratio=0.5):
    return _planning_method_instructions('both', features_per_round, ratio)


def test_single_method_runs_keep_their_restriction():
    assert "only choose the 'code' method" in _planning_method_instructions('code', 5, 0.5)
    assert "only choose the 'vlm' method" in _planning_method_instructions('vlm', 5, 0.5)


@pytest.mark.parametrize('per_round,ratio,code,vlm', [
    (10, 0.5, 5, 5),
    (5, 0.5, 2, 3),
    (2, 0.5, 1, 1),
    (6, 0.8, 5, 1),   # clamped so the minority method never disappears
    (6, 0.1, 1, 5),
])
def test_mixed_runs_request_a_split_that_keeps_both_methods(per_round, ratio, code, vlm):
    text = instructions(per_round, ratio)
    assert f'roughly {code} feature(s) with "method": "code"' in text
    assert f'{vlm} feature(s) with "method": "vlm"' in text
    assert 'each method must appear at least once' in text


def test_a_single_feature_cannot_be_split():
    assert instructions(1) == 'None'


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
    assert 'roughly 5 feature(s) with "method": "code"' in prompt
    assert '{method_instructions}' not in prompt
