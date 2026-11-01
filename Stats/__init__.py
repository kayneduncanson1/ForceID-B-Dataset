import numpy as np
import pandas as pd


def extract_grf_features(signals, feature_specs):

    """Extract pre-specified Fz/Fy features from signals in %BW.

    Returns an array (n_signals, n_features). Each window is defined using
    an explicit 0-100% stance time axis rather than assuming a particular
    integer index is exactly the stated percentage.
    """

    values = np.asarray(signals, dtype=float)

    if values.ndim != 3 or values.shape[1] != 5:
        raise ValueError("Expected signals with shape (n_signals, 5, n_frames).")

    time_pct = np.linspace(0.0, 100.0, values.shape[2])

    feature_values = []

    for spec in feature_specs:

        mask = (time_pct >= spec["start_pct"]) & (time_pct <= spec["end_pct"])

        if not np.any(mask):
            raise ValueError(f"Empty time window for feature {spec['key']}.")

        window = values[:, spec["channel"], :][:, mask]

        if spec["operation"] == "max":
            feature_values.append(np.max(window, axis=1))
        else:
            feature_values.append(np.min(window, axis=1))

    result = np.column_stack(feature_values)

    if not np.all(np.isfinite(result)):
        bad_rows = np.flatnonzero(~np.all(np.isfinite(result), axis=1))
        raise ValueError(f"Non-finite GRF feature values at signal indices {bad_rows[:10].tolist()}.")

    return result


def aggregate_participant_features(features, participant_labels, feature_keys):

    """Average bilateral trial-level features across all trials within person.

    Each input row is one trial-level bilateral summary (right/left stance
    feature values have already been averaged). Trials from all available
    sessions are pooled and averaged directly within participant, consistently
    across ForceID-B_CFC and all benchmark datasets. Thus, participants are the
    independent units in the between-dataset comparison, while each participant's
    observed trials contribute equally to that participant's own mean.
    """

    df = pd.DataFrame(features, columns=feature_keys)
    df["_participant"] = [str(label) for label in participant_labels]
    participant_means = df.groupby("_participant", sort=True)[feature_keys].mean()
    participant_means.index.name = "participant"

    if participant_means.isna().any().any():
        raise ValueError("NaN participant-level feature values encountered.")

    return participant_means


def participant_mass_map_from_fib_metadata(metadata):

    """Map each ForceID-B participant to mean body mass (without objects) across sessions. Metadata rows are
    trial-level, so first collapse repeated rows within sessions, then give each observed session equal weight.
    Column index 1 is participant ID, column index 2 is session and column index 7 is body mass without carried
    objects."""

    id_values = np.asarray(metadata[:, 1])
    session_values = np.asarray(metadata[:, 2])
    mass_values = pd.to_numeric(pd.Series(metadata[:, 7]), errors="coerce").to_numpy(dtype=float)

    df = pd.DataFrame({
        "_participant": [str(value) for value in id_values],
        "_session": [str(value) for value in session_values],
        "_mass": mass_values,
    })

    session_means = df.groupby(["_participant", "_session"], sort=True)["_mass"].mean()
    participant_means = session_means.groupby(level="_participant", sort=True).mean().dropna()

    return {str(pid): float(mass) for pid, mass in participant_means.items()}


def hedges_g(x, y):

    """Hedges' g for independent participant-level summaries (x minus y)."""

    n_x, n_y = x.size, y.size

    if n_x < 2 or n_y < 2:
        return np.nan

    degrees_freedom = n_x + n_y - 2

    pooled_variance = (
        (n_x - 1) * np.var(x, ddof=1) + (n_y - 1) * np.var(y, ddof=1)
    ) / degrees_freedom

    pooled_sd = np.sqrt(pooled_variance)

    if not np.isfinite(pooled_sd) or pooled_sd <= 0:
        return np.nan

    correction = 1.0 - 3.0 / (4.0 * degrees_freedom - 1.0)

    return correction * (np.mean(x) - np.mean(y)) / pooled_sd


def bootstrap_difference_and_g(x, y, n_bootstrap, seed, batch_size=500):

    """Estimate differences and percentile CIs by independently resampling people.

    Each row in x/y is one participant, so clustering is handled before this
    function is called. Batching avoids allocating a very large bootstrap
    matrix. Returns point estimates and 95% percentile confidence intervals
    for mean difference (x-y) and Hedges' g.
    """

    if x.size < 2 or y.size < 2:
        raise ValueError("At least two participants per dataset are required for effect sizes.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite participant-level values found in effect size calculation.")

    diff_point = float(np.mean(x) - np.mean(y))
    g_point = float(hedges_g(x, y))

    if not np.isfinite(g_point):
        raise ValueError("Hedges' g is undefined because pooled participant-level SD is zero.")

    rng = np.random.default_rng(seed)
    boot_diff = np.empty(n_bootstrap, dtype=float)
    boot_g = np.empty(n_bootstrap, dtype=float)
    out_i = 0
    while out_i < n_bootstrap:
        n_batch = min(batch_size, n_bootstrap - out_i)
        xb = x[rng.integers(0, x.size, size=(n_batch, x.size))]
        yb = y[rng.integers(0, y.size, size=(n_batch, y.size))]
        mean_x = np.mean(xb, axis=1)
        mean_y = np.mean(yb, axis=1)
        diff_b = mean_x - mean_y
        var_x = np.var(xb, axis=1, ddof=1)
        var_y = np.var(yb, axis=1, ddof=1)
        df = x.size + y.size - 2
        pooled_sd = np.sqrt(((x.size - 1) * var_x + (y.size - 1) * var_y) / df)
        correction = 1.0 - 3.0 / (4.0 * df - 1.0)
        g_b = correction * diff_b / pooled_sd
        # A zero pooled SD is theoretically possible in a bootstrap replicate, so mark as NaN and omit when obtaining
        # the percentile interval.
        g_b[~np.isfinite(g_b)] = np.nan
        boot_diff[out_i:out_i + n_batch] = diff_b
        boot_g[out_i:out_i + n_batch] = g_b
        out_i += n_batch

    diff_ci = np.nanquantile(boot_diff, [0.025, 0.975])
    valid_g = boot_g[np.isfinite(boot_g)]

    if valid_g.size < 0.95 * n_bootstrap:
        raise RuntimeError("Too many undefined bootstrap Hedges' g replicates. Inspect feature variability.")

    g_ci = np.quantile(valid_g, [0.025, 0.975])

    return diff_point, diff_ci, g_point, g_ci


def print_feature_sign_diagnostics(dataset_name, trial_features, feature_keys):

    """Print simple sign convention checks."""

    braking_negative = np.mean(trial_features[:, feature_keys.index("Fy_braking_minimum")] < 0) * 100.0
    propulsion_positive = np.mean(trial_features[:, feature_keys.index("Fy_propulsive_maximum")] > 0) * 100.0
    fz_positive = np.mean(np.all(trial_features[:, :3] > 0, axis=1)) * 100.0

    print(
        f"Sign check {dataset_name}: braking minimum < 0 in {braking_negative:.1f}% of trials; "
        f"propulsive maximum > 0 in {propulsion_positive:.1f}%; all three Fz features > 0 in {fz_positive:.1f}%."
    )

    if braking_negative < 80.0 or propulsion_positive < 80.0:
        print(
            f"WARNING: {dataset_name} has an atypical Fy sign distribution. Verify coordinate/sign "
            "conventions and inspect waveforms before interpreting feature results."
        )
