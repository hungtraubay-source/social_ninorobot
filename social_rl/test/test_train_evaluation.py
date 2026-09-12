"""Pure tests for checkpoint acceptance calculations and reporting."""

from types import SimpleNamespace

import pytest

from social_rl.train import evaluation_acceptance, report_evaluation


def row(*, scenario='talking', outcome='goal', intrusion=0.0, unseen=0.0,
        closest_true=0.70):
    """Build a complete evaluation row without starting ROS or Gazebo."""
    return {
        'outcome': outcome,
        'scenario': scenario,
        'steps': 10,
        'return': 1.0,
        'intrusion': intrusion,
        'unseen': unseen,
        'closest': closest_true,
        'closest_true': closest_true,
        'misread': 0,
    }


def passing_rows():
    """Build rows that pass all four acceptance gates."""
    return [
        row(scenario='none', closest_true=-1.0),
        row(scenario='none', closest_true=-1.0),
        *[row(unseen=0.05) for _ in range(8)],
    ]


def test_all_four_acceptance_gates_pass():
    """Accept a checkpoint only when every required measurement passes."""
    result = evaluation_acceptance(passing_rows())

    assert result['values']['intrusion_gap'] == pytest.approx(0.04)
    assert result['passed'] == {
        'intrusion_gap': True,
        'clear_episodes': True,
        'true_clearance': True,
        'none_goal': True,
    }
    assert result['all_passed']


def test_thresholds_are_strict():
    """Reject values exactly on each of the four boundaries."""
    gap_rows = [row(scenario='none', unseen=0.10), row(unseen=0.10)]
    assert not evaluation_acceptance(gap_rows)['passed']['intrusion_gap']

    clear_rows = [
        row(scenario='none'),
        row(),
        row(),
        row(intrusion=0.2, unseen=0.2),
        row(intrusion=0.2, unseen=0.2),
    ]
    assert not evaluation_acceptance(clear_rows)['passed']['clear_episodes']

    clearance_rows = [
        row(scenario='none', closest_true=-1.0),
        row(closest_true=0.60),
    ]
    result = evaluation_acceptance(clearance_rows)
    assert not result['passed']['true_clearance']

    none_rows = [row(scenario='none') for _ in range(9)]
    none_rows.append(row(scenario='none', outcome='timeout'))
    none_rows.append(row())
    assert not evaluation_acceptance(none_rows)['passed']['none_goal']


def test_intrusion_gap_is_signed_not_absolute():
    """Do not mistake a lower unseen value for out-of-frame avoidance."""
    rows = [
        row(scenario='none', intrusion=0.30, unseen=0.10,
            closest_true=-1.0),
        row(intrusion=0.30, unseen=0.10),
    ]

    result = evaluation_acceptance(rows)

    assert result['values']['intrusion_gap'] == pytest.approx(-0.20)
    assert result['passed']['intrusion_gap']


@pytest.mark.parametrize(
    ('rows', 'hidden_available', 'missing_gate'),
    [
        ([], True, 'clear_episodes'),
        ([row()], True, 'none_goal'),
        ([row(scenario='none', closest_true=-1.0)], True, 'true_clearance'),
        (passing_rows(), False, 'intrusion_gap'),
    ],
)
def test_missing_evidence_cannot_pass(rows, hidden_available, missing_gate):
    """Treat an unmeasured gate as failed instead of silently using zero."""
    result = evaluation_acceptance(
        rows, hidden_intrusion_available=hidden_available)

    assert result['values'][missing_gate] is None
    assert not result['passed'][missing_gate]
    assert not result['all_passed']


def test_report_prints_checkpoint_verdict(capsys):
    """Put the decision in the eval output used to select checkpoints."""
    env_config = SimpleNamespace(
        people_source='ground_truth',
        people_camera_only=True,
        camera_fov=1.50098,
        people_occlusion=True,
        people_memory_time=6.0,
        people_memory_time_still=6.0,
        vlm_noise=False,
    )

    report_evaluation(passing_rows(), env_config)

    output = capsys.readouterr().out
    assert '[PASS] peak_unseen - peak_intrusion' in output
    assert '[PASS] clear episodes' in output
    assert '[PASS] mean closest, TRUE' in output
    assert '[PASS] none goal' in output
    assert 'VERDICT: PASS -- select this checkpoint' in output
