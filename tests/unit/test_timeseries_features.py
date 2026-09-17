from __future__ import annotations

import numpy as np
import pandas as pd

from aegis_ai.features.timeseries import (
    CausalRollingFeatureConfig,
    build_causal_rolling_features,
    finite_feature_mask,
)


def test_causal_rolling_features_use_past_values_only() -> None:
    frame = pd.DataFrame({"metric_value": [1.0, 2.0, 10.0, 4.0]})
    features = build_causal_rolling_features(
        frame,
        config=CausalRollingFeatureConfig(
            short_window=2,
            long_window=3,
            trend_window=2,
            min_periods=1,
        ),
    )

    assert features.loc[2, "rolling_mean_short"] == 1.5
    assert features.loc[2, "diff_1"] == 8.0
    assert np.isclose(features.loc[2, "pct_change_1"], 4.0)


def test_finite_feature_mask_excludes_warmup_rows() -> None:
    frame = pd.DataFrame({"metric_value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    features = build_causal_rolling_features(
        frame,
        config=CausalRollingFeatureConfig(
            short_window=2,
            long_window=2,
            trend_window=2,
            min_periods=2,
        ),
    )

    mask = finite_feature_mask(features)

    assert mask.tolist() == [False, False, True, True, True]
