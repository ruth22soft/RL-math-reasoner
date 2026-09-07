from pathlib import Path

import generate_chapters_pdf as gcp


def test_beta_trajectory_supports_multiple_controllers():
    curves = gcp._controller_beta_curves()
    assert set(curves) >= {"fixed", "rule", "mlp", "lstm"}
    for name, series in curves.items():
        assert len(series["steps"]) > 0
        assert len(series["beta"]) == len(series["steps"])
