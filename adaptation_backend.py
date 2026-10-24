import csv
import json
import numpy as np

from bioscrape.types import Model
from bioscrape.simulator import py_simulate_model


SPECIES = ["A", "B", "C", "L"]

PARAMETER_NAMES = [
    "k1", "k2", "k3", "k4", "k5", "k6",
    "KIA", "KFA", "KB", "KFB", "KAC", "KBC"
]

WARMUP = 500.0
DURATION = 300.0
FINAL_WINDOW = 20.0
REPORT_DT = 0.2

ERROR_LIMIT = 0.02
SPAN_LIMIT = 0.002

MC_DRAWS = 20
MC_SEED = 2026

PERTURB_LOW = 0.8
PERTURB_HIGH = 1.25

sweep_rows = []


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def validate(mode, parameters, initial):
    if mode not in ("NFBLB", "IFFLP"):
        raise ValueError("Mode must be NFBLB or IFFLP")

    if len(parameters) != 12 or min(parameters) <= 0:
        raise ValueError("All 12 parameters must be positive")

    if len(initial) != 3:
        raise ValueError("Three initial activities are required")

    if not all(0 <= x <= 1 for x in initial):
        raise ValueError("Initial activities must lie in [0,1]")


def derivative(y, mode, parameters, ligand):
    a, b, c = y

    (
        k1, k2, k3, k4, k5, k6,
        kia, kfa, kb, kfb, kac, kbc
    ) = parameters

    driver = c if mode == "NFBLB" else a

    return np.array([
        ligand * k1 * (1-a) / (1-a+kia)
        - k2 * a / (a+kfa),

        driver * k3 * (1-b) / (1-b+kb)
        - k4 * b / (b+kfb),

        a * k5 * (1-c) / (1-c+kac)
        - b * k6 * c / (c+kbc)
    ])


def make_model(mode, parameters, initial, ligand):
    validate(mode, parameters, initial)

    if ligand <= 0:
        raise ValueError("Ligand must be positive")

    driver = "C" if mode == "NFBLB" else "A"

    rates = [
        ("A", True, "L*k1*(1-A)/(1-A+KIA)"),
        ("A", False, "k2*A/(A+KFA)"),

        ("B", True, driver + "*k3*(1-B)/(1-B+KB)"),
        ("B", False, "k4*B/(B+KFB)"),

        ("C", True, "A*k5*(1-C)/(1-C+KAC)"),
        ("C", False, "B*k6*C/(C+KBC)")
    ]

    reactions = []

    for species, positive, rate in rates:
        if positive:
            reactions.append(
                ([], [species], "general", {"rate": rate})
            )
        else:
            reactions.append(
                ([species], [], "general", {"rate": rate})
            )

    values = list(map(float, initial)) + [float(ligand)]

    return Model(
        species=SPECIES,
        reactions=reactions,
        parameters=list(
            zip(PARAMETER_NAMES, map(float, parameters))
        ),
        initial_condition_dict=dict(zip(SPECIES, values))
    )


def simulate(model, values, times):
    model.set_species(
        dict(zip(SPECIES, map(float, values)))
    )

    result = py_simulate_model(
        timepoints=np.asarray(times, dtype=float),
        Model=model,
        stochastic=False
    )

    data = result[SPECIES].to_numpy(dtype=float)

    if not np.isfinite(data).all():
        raise FloatingPointError("Nonfinite simulation result")

    if (
        data[:, :3].min() < -1e-5
        or data[:, :3].max() > 1.00001
    ):
        raise FloatingPointError(
            "Activity outside [0,1]. Check solver/settings."
        )

    return data


def is_settled(values, mode, parameters, ligand):
    residual = derivative(
        values[:3], mode, parameters, ligand
    )

    return bool(np.max(np.abs(residual)) < 1e-6)


# ============================================================
# SCORING
# ============================================================

def metrics(times, outputs, baseline, old_input, new_input,
            valid_baseline, complete):
    times = np.asarray(times, dtype=float)
    outputs = np.asarray(outputs, dtype=float)

    scale = max(abs(baseline), 1e-12)

    input_change = abs(new_input-old_input) / old_input

    relative = np.abs(outputs-baseline) / scale
    peak = float(relative.max())

    tail_mask = times >= max(
        0.0, float(times[-1])-FINAL_WINDOW
    )

    tail = outputs[tail_mask]

    error = float(
        np.max(np.abs(tail-baseline)) / scale
    )

    span = float(np.ptp(tail) / scale)

    sensitivity = peak / max(input_change, 1e-12)

    precision = min(
        1e9,
        input_change / max(error, 1e-12)
    )

    recovery = -1.0
    outside = np.where(relative > ERROR_LIMIT)[0]

    if len(outside) and relative[-1] <= ERROR_LIMIT:
        next_index = int(outside[-1]) + 1

        if next_index < len(times):
            candidate = float(times[next_index])

            if times[-1]-candidate >= FINAL_WINDOW:
                recovery = candidate

    failures = []

    if not valid_baseline:
        failures.append("unsettled baseline")

    if peak <= 0.01:
        failures.append("peak <= 1%")

    if sensitivity <= 1:
        failures.append("S <= 1")

    if precision <= 10:
        failures.append("P <= 10")

    if error > ERROR_LIMIT:
        failures.append("recovery error > 2%")

    if span > SPAN_LIMIT:
        failures.append("tail variation > 0.2%")

    passed = len(failures) == 0

    status = "RUNNING: metrics provisional"

    if complete:
        if passed:
            status = "PASS: sensitive near-adaptation"
        else:
            status = "FAIL: " + "; ".join(failures)

    return {
        "error": error * 100,
        "peak": peak * 100,
        "S": float(sensitivity),
        "P": float(precision),
        "recovery": recovery,
        "passed": bool(passed),
        "status": status,
        "reason": "PASS" if passed else "; ".join(failures)
    }


# ============================================================
# INTERACTIVE SIMULATION
# ============================================================

def initialize(mode, parameters, initial):
    global live_model, live_state, live_config
    global clock, baseline, baseline_valid
    global event_time, old_ligand, history

    parameters = list(map(float, parameters))
    initial = list(map(float, initial))

    validate(mode, parameters, initial)

    live_config = (mode, parameters)

    live_model = make_model(
        mode, parameters, initial, 1.0
    )

    live_state = simulate(
        live_model,
        initial + [1.0],
        [0.0, WARMUP]
    )[-1]

    baseline = float(live_state[2])

    if baseline <= 1e-8:
        raise ValueError(
            "Baseline C too small for relative metrics"
        )

    baseline_valid = is_settled(
        live_state, mode, parameters, 1.0
    )

    clock = 0.0
    event_time = None
    old_ligand = 1.0
    history = []


def apply_stimulus(level):
    global live_state, baseline, baseline_valid
    global event_time, old_ligand, history

    level = float(level)

    if level <= 0:
        raise ValueError("Stimulus must be positive")

    if abs(level-float(live_state[3])) < 1e-12:
        raise ValueError("Choose a different stimulus level")

    mode, parameters = live_config

    old_ligand = float(live_state[3])

    baseline_valid = is_settled(
        live_state, mode, parameters, old_ligand
    )

    baseline = float(live_state[2])

    if baseline <= 1e-8:
        raise ValueError("Baseline too small")

    event_time = float(clock)

    # Only the external input changes immediately.
    live_state[3] = level

    history = [(0.0, baseline)]


def finished():
    return bool(
        event_time is not None
        and clock-event_time >= DURATION-1e-9
    )


def advance():
    global live_state, clock

    if finished():
        return

    step = REPORT_DT

    if event_time is not None:
        step = min(
            step,
            DURATION-(clock-event_time)
        )

    live_state = simulate(
        live_model,
        live_state,
        [0.0, step]
    )[-1]

    clock += step

    if event_time is not None:
        history.append(
            (float(clock-event_time),
             float(live_state[2]))
        )


def export_state():
    if event_time is None:
        result = {
            "error": 0.0,
            "peak": 0.0,
            "S": 0.0,
            "P": 0.0,
            "recovery": -1.0,
            "status": "WAITING: apply stimulus"
        }
    else:
        data = np.asarray(history)

        result = metrics(
            data[:, 0],
            data[:, 1],
            baseline,
            old_ligand,
            float(live_state[3]),
            baseline_valid,
            finished()
        )

    return (
        [float(clock)]
        + list(map(float, live_state))
        + [
            float(baseline),
            bool(baseline_valid),
            result["error"],
            result["peak"],
            result["S"],
            result["P"],
            result["recovery"],
            result["status"],
            finished()
        ]
    )


# ============================================================
# INDEPENDENT BATCH TRIALS
# ============================================================

def independent_trial(mode, parameters, initial, new_input):
    trial_model = make_model(
        mode, parameters, initial, 1.0
    )

    values = simulate(
        trial_model,
        list(initial) + [1.0],
        [0.0, WARMUP]
    )[-1]

    valid = is_settled(
        values, mode, parameters, 1.0
    )

    base = float(values[2])

    if base <= 1e-8:
        return {
            "error": 100.0,
            "peak": 0.0,
            "S": 0.0,
            "P": 0.0,
            "recovery": -1.0,
            "passed": False,
            "reason": "baseline too small"
        }, False

    values[3] = new_input

    times = np.linspace(
        0.0,
        DURATION,
        int(DURATION/REPORT_DT)+1
    )

    trajectory = simulate(
        trial_model, values, times
    )

    result = metrics(
        times,
        trajectory[:, 2],
        base,
        1.0,
        new_input,
        valid,
        True
    )

    return result, valid


def run_batch(mode, parameters, initial):
    global sweep_rows

    parameters = list(map(float, parameters))
    initial = list(map(float, initial))

    validate(mode, parameters, initial)

    original = np.asarray(parameters, dtype=float)

    # Reusing the seed allows reproducible comparisons.
    rng = np.random.default_rng(MC_SEED)

    candidates = [("Reference", original.copy())]

    for index in range(MC_DRAWS):
        factors = np.exp(
            rng.uniform(
                np.log(PERTURB_LOW),
                np.log(PERTURB_HIGH),
                len(original)
            )
        )

        candidates.append(
            ("Draw " + str(index+1), original*factors)
        )

    sweep_rows = []

    passed = 0
    tested = 0
    reference_passed = 0

    for label, candidate in candidates:
        for ligand in [1.2, 1.5]:
            result, valid = independent_trial(
                mode,
                candidate.tolist(),
                initial,
                ligand
            )

            sweep_rows.append([
                label,
                float(ligand),
                result["error"],
                result["peak"],
                result["S"],
                result["P"],
                result["recovery"],
                bool(valid),
                bool(result["passed"]),
                candidate.tolist(),
                result["reason"]
            ])

            if label == "Reference":
                reference_passed += int(result["passed"])
            else:
                tested += 1
                passed += int(result["passed"])

    return [
        sweep_rows,
        float(100.0 * passed/tested),
        int(passed),
        int(tested),
        int(reference_passed)
    ]


def save_sweep(path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "test",
            "post_input",
            "tail_max_error_pct",
            "peak_deviation_pct",
            "sensitivity",
            "conservative_precision",
            "recovery_time",
            "baseline_settled",
            "pass",
            "parameters_json",
            "reason"
        ])

        for row in sweep_rows:
            writer.writerow(
                row[:9]
                + [json.dumps(row[9]), row[10]]
            )