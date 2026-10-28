"""QAOA for the panel QUBO, written with Qiskit.

The QUBO matrix is turned into a `QuadraticProgram` and then into an Ising operator with
qiskit_optimization. The circuit is `QAOAAnsatz` with `reps` layers, run on the Aer
statevector simulator through the Aer `SamplerV2`, with a finite number of shots per circuit
evaluation. COBYLA from qiskit_algorithms tunes the angles. The objective that COBYLA minimises
is the mean QUBO energy of the sampled bit strings.

Bit order: Qiskit prints bit strings with qubit 0 on the right, so int(bitstring, 2) is the
state index used in qbio.qubo (bit i of the index is variable i).
"""
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram

from . import qubo

_BACKEND = AerSimulator(method="statevector")


def to_quadratic_program(Q):
    n = Q.shape[0]
    qp = QuadraticProgram("gene_panel")
    for i in range(n):
        qp.binary_var(f"x{i}")
    qp.minimize(
        linear={f"x{i}": float(Q[i, i]) for i in range(n)},
        quadratic={(f"x{i}", f"x{j}"): float(Q[i, j]) for i in range(n) for j in range(i + 1, n)},
    )
    return qp


def build_circuit(Q, reps):
    """Transpiled QAOA circuit with measurements. The cost operator is scaled to unit size."""
    op, _offset = to_quadratic_program(Q).to_ising()
    scale = float(np.abs(op.coeffs).max())
    ansatz = QAOAAnsatz(cost_operator=op / scale, reps=reps)
    ansatz.measure_all()
    return transpile(ansatz, _BACKEND, optimization_level=1), scale


def initial_point(reps, rng, gamma_max=0.9, beta_max=0.8, jitter=0.1):
    """Linear ramp start plus seeded jitter.

    Cost angles grow and mixer angles shrink across the layers. For minimisation with the
    QAOAAnsatz convention the mixer angles are negative, which a noise free scan of depth 1
    on the full pool problem confirmed (best beta -0.39, gamma 0.47).
    """
    layer = (np.arange(reps) + 0.5) / reps
    gammas = gamma_max * layer
    betas = -beta_max * (1 - layer)
    point = np.concatenate([betas, gammas])  # QAOAAnsatz orders beta parameters, then gamma
    return point + rng.normal(0, jitter, size=point.shape)


def _mean_energy(counts, E):
    total = 0
    acc = 0.0
    for bits, c in counts.items():
        acc += c * E[int(bits, 2)]
        total += c
    return acc / total


def run_qaoa(Q, E, reps, seed, maxiter=150, shots=1024, final_shots=8192, rhobeg=0.3):
    """One QAOA run. E is the energy of every state (from qubo.energies) used for scoring."""
    start = time.perf_counter()
    circuit, _scale = build_circuit(Q, reps)
    rng = np.random.default_rng(seed)
    sampler = SamplerV2(default_shots=shots, seed=seed)
    runs = {"n": 0}

    def objective(theta):
        runs["n"] += 1
        counts = sampler.run([(circuit, theta)]).result()[0].data.meas.get_counts()
        return _mean_energy(counts, E)

    opt = COBYLA(maxiter=maxiter, rhobeg=rhobeg)
    result = opt.minimize(objective, initial_point(reps, rng))

    final_sampler = SamplerV2(default_shots=final_shots, seed=seed + 10_000)
    counts = final_sampler.run([(circuit, result.x)]).result()[0].data.meas.get_counts()
    runtime = time.perf_counter() - start
    return {
        "reps": reps,
        "seed": seed,
        "theta": result.x,
        "counts": counts,
        "circuit_runs": runs["n"] + 1,
        "shots_total": runs["n"] * shots + final_shots,
        "runtime_s": runtime,
        "final_mean_energy": _mean_energy(counts, E),
    }


def summarise(run, E, sizes, k, optimum_states):
    """Turn raw counts into the quantities the task asks for."""
    counts = run["counts"]
    total = sum(counts.values())
    states = np.array([int(b, 2) for b in counts])
    freq = np.array([counts[b] for b in counts], dtype=float) / total
    size = sizes[states]
    right = size == k
    opt_set = set(int(s) for s in optimum_states)
    prob_opt = float(sum(f for s, f in zip(states, freq) if int(s) in opt_set))
    top = int(states[np.argmax(freq)])
    best_valid = None
    if right.any():
        cand = states[right]
        best_valid = int(cand[np.argmin(E[cand])])
    return {
        "prob_right_size": float(freq[right].sum()),
        "prob_optimum": prob_opt,
        "most_probable_state": top,
        "most_probable_size": int(sizes[top]),
        "most_probable_is_right_size": bool(sizes[top] == k),
        "best_sampled_valid_state": best_valid,
        "found_optimum": bool(best_valid is not None and best_valid in opt_set),
        "distinct_states_sampled": int(len(states)),
    }
