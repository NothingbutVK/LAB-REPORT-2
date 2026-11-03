
# app.py
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- Fixed GA Parameters (assignment requirements) ----
POP_SIZE = 300
CHROM_LEN = 80
TARGET_ONES = 50
MAX_FITNESS = 80
N_GENERATIONS = 50

# ---- GA Hyperparameters ----
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN

# ---- Fitness Function ----
def fitness(individual: np.ndarray) -> float:
    """
    Fitness reaches its maximum value of 80
    when the number of ones equals 50.
    """
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)


# ---- GA Operators ----
def init_population(pop_size: int, chrom_len: int) -> np.ndarray:
    return np.random.randint(
        0, 2, size=(pop_size, chrom_len), dtype=np.int8
    )


def tournament_selection(
    pop: np.ndarray, fits: np.ndarray, k: int
) -> np.ndarray:
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()


def single_point_crossover(p1: np.ndarray, p2: np.ndarray):
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()

    point = np.random.randint(1, CHROM_LEN)

    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])

    return c1, c2


def mutate(individual: np.ndarray) -> np.ndarray:
    individual = individual.copy()
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    individual[mask] = 1 - individual[mask]
    return individual


# ---- Genetic Algorithm ----
def evolve(pop: np.ndarray, generations: int):
    best_fitness_per_gen = []
    best_individual = None
    best_f = -np.inf

    for _ in range(generations):
        fits = np.array([fitness(ind) for ind in pop])

        # Track the best individual in this generation
        gen_best_idx = np.argmax(fits)
        gen_best = pop[gen_best_idx]
        gen_best_f = float(fits[gen_best_idx])

        if gen_best_f > best_f:
            best_f = gen_best_f
            best_individual = gen_best.copy()

        best_fitness_per_gen.append(best_f)

        # Stop if the maximum fitness is achieved
        if best_f == MAX_FITNESS:
            break

        # Create the next generation
        new_pop = []

        while len(new_pop) < len(pop):
            p1 = tournament_selection(pop, fits, TOURNAMENT_K)
            p2 = tournament_selection(pop, fits, TOURNAMENT_K)

            c1, c2 = single_point_crossover(p1, p2)

            c1 = mutate(c1)
            c2 = mutate(c2)

            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:len(pop)], dtype=np.int8)

    return best_individual, best_f, best_fitness_per_gen


# ---- Streamlit UI ----
st.set_page_config(
    page_title="Genetic Algorithm",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Genetic Algorithm: Bit Pattern")
st.caption(
    "Search Algorithms | Genetic Algorithm"
)

with st.expander("ℹ️ Problem Setup", expanded=True):
    st.write(
        f"""
- **Population size:** `{POP_SIZE}`
- **Chromosome length:** `{CHROM_LEN}`
- **Target number of ones:** `{TARGET_ONES}`
- **Maximum fitness:** `{MAX_FITNESS}`
- **Number of generations:** `{N_GENERATIONS}`
- **Selection:** Tournament Selection (k={TOURNAMENT_K})
- **Crossover:** Single-point Crossover
- **Mutation:** Bit-flip Mutation
"""
    )

col1, col2 = st.columns(2)

with col1:
    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1
    )

with col2:
    run_btn = st.button(
        "Run Genetic Algorithm",
        type="primary"
    )

if run_btn:
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Initialisation
    population = init_population(POP_SIZE, CHROM_LEN)

    # Run the algorithm
    best_ind, best_fit, curve = evolve(
        population, N_GENERATIONS
    )

    # Results
    ones_count = int(best_ind.sum())
    zeros_count = CHROM_LEN - ones_count
    bitstring = "".join(map(str, best_ind.tolist()))

    st.subheader("🏁 Best Individual Found")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best Fitness",
        f"{best_fit:.0f}/{MAX_FITNESS}"
    )
    col2.metric("Number of Ones", ones_count)
    col3.metric("Number of Zeros", zeros_count)

    st.write(f"**Chromosome Length:** {len(best_ind)} bits")
    st.code(bitstring, language="text")

    # Fitness calculation
    st.write("### Fitness Calculation")
    st.latex(
        r"f(x)=80-|N-50|"
    )
    st.write(
        f"Fitness = 80 - |{ones_count} - 50| "
        f"= {fitness(best_ind):.0f}"
    )

    # Convergence plot
    st.subheader("📈 Fitness Progress")

    fig, ax = plt.subplots()

    ax.plot(
        range(1, len(curve) + 1),
        curve,
        marker="o",
        linewidth=2
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Genetic Algorithm Convergence")
    ax.set_ylim(0, MAX_FITNESS + 5)
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)

    # Validation
    if best_fit == MAX_FITNESS and ones_count == TARGET_ONES:
        st.success(
            "Maximum fitness achieved! "
            "Number of ones = 50 and fitness = 80."
        )
    else:
        st.info(
            "The target was not reached in this run. "
            "Try another random seed."
        )

st.caption("Genetic Algorithm | Population = 300 | 80-bit chromosome")
