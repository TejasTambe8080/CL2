"""
Experiment 5 : Tarrataca's quantum production system with the 3-puzzle (Grover over production-rule sequences)
=======================================================================================================================
INSTRUCTIONS
 1. Install:  pip install qiskit qiskit-aer matplotlib numpy
 2. Dataset : None. The only input is the demo START and GOAL boards of the 2x2 puzzle, typed in Step 2.
              Runs on the local AerSimulator (no IBM account or API token needed).
 3. Run     : python QAI_3_Tarrataca_3Puzzle.py
 4. Steps   : boards + rules -> rules as unitary matrices -> oracle (run rules, test goal, undo rules)
              -> diffuser -> Grover loop -> measure rule qubits -> replay the winning sequence
"""

# --- Step 1: Import the libraries ---
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

# --- Step 2: The 3-puzzle (2x2 board), production rules and demo input ---
# Board = tuple of 4 numbers read row by row:
# (top-left, top-right, bottom-left, bottom-right), 0 = blank.
GOAL = (1, 2, 3, 0)           # DEMO INPUT: goal board
START = (2, 0, 1, 3)          # DEMO INPUT: start board
STEPS = 3                     # how many rules we apply one after the other
# all 24 boards, the position in this list is the board number (fits in 5 qubits)
BOARDS = list(itertools.permutations(range(4)))

# Draw a board as two lines of text.
def show(board):
    for row in (board[:2], board[2:]):
        symbols = []
        for t in row:
            if t:
                symbols.append(str(t))
            else:
                symbols.append("_")
        print("   " + " ".join(symbols))

# Production rule: V swaps the blank with its vertical neighbour,
# H swaps it with its horizontal neighbour.
def apply_rule(board, rule):
    b = list(board)
    i = b.index(0)
    # on a 2x2 board both neighbours always exist
    if rule == "V":
        j = i ^ 2
    else:
        j = i ^ 1
    b[i], b[j] = b[j], b[i]
    return tuple(b)

print("Start board (demo input):")
show(START)
print("Goal board (demo input):")
show(GOAL)

# --- Step 3: Write the rule table as a unitary matrix (classical preparation) ---
# Gate acts on 6 qubits: 1 rule qubit (lowest) + 5 board qubits.
# Rule qubit 0 = V, 1 = H.
# It maps |rule>|board> to |rule>|rule applied to board>.
# Boards 24..31 do not exist, so they stay unchanged.
U = np.zeros((64, 64))
for x in range(64):
    r, b = x & 1, x >> 1
    if b < 24:
        new_board = apply_rule(BOARDS[b], "VH"[r])
        nb = BOARDS.index(new_board)
    else:
        nb = b
    # one 1 per column: a permutation matrix, so it is unitary
    U[r + 2 * nb, x] = 1
# every rule undoes itself, so applying it again = uncompute
assert np.allclose(U @ U, np.eye(64))
# qiskit also checks that U is unitary here
RULE = UnitaryGate(U, label="Rule")

# --- Step 4: Count the marked sequences (to choose the Grover iterations) ---
# Grover needs M (how many sequences reach the goal) to pick the iteration count.
# We find M by trying all 8 sequences classically.
# The oracle itself does NOT contain the answers.
N = 2 ** STEPS
def name(k):
    # number k -> rule sequence, step 1 = lowest bit
    seq = ""
    for s in range(STEPS):
        seq = seq + "VH"[(k >> s) & 1]
    return seq
def replay(seq):
    board = START
    for rule in seq:
        board = apply_rule(board, rule)
    return board
# also used to read the result (exact probability)
marked = []
for k in range(N):
    if replay(name(k)) == GOAL:
        marked.append(k)
M = len(marked)
assert M > 0, (
    "No sequence of STEPS rules reaches the goal - "
    "change STEPS, START or GOAL"
)
print("\nSequences N =", N, " marked M =", M)

# --- Step 5: Build the oracle: run the rules, test the goal, undo the rules ---
# rule qubits: one per step (the sequence lives here)
RQ = [0, 1, 2]
# board register: 5 qubits holding the board number
BQ = [3, 4, 5, 6, 7]
# the goal test compares the board with this number
goal_no = BOARDS.index(GOAL)

def oracle():
    qc = QuantumCircuit(8, name="Oracle")
    # apply the chosen rule of each step to the board
    for s in RQ:
        qc.append(RULE, [s] + BQ)
    zeros = []
    for q in range(5):
        if not (goal_no >> q) & 1:
            zeros.append(BQ[q])
    # phase flip (-1) if board == GOAL
    qc.x(zeros)
    qc.h(BQ[4])
    qc.mcx(BQ[:4], BQ[4])
    qc.h(BQ[4])
    qc.x(zeros)
    # undo the rules: the board is back to START
    for s in reversed(RQ):
        qc.append(RULE, [s] + BQ)
    return qc

print("\nOracle circuit (rules, goal test, undo):")
print(oracle().draw(output="text", fold=100))

# --- Step 6: Build the diffuser (inversion about the average) on the rule qubits ---
def diffuser():
    qc = QuantumCircuit(8, name="Diffuser")
    qc.h(RQ)
    qc.x(RQ)
    qc.h(RQ[-1])
    qc.mcx(RQ[:-1], RQ[-1])
    qc.h(RQ[-1])
    qc.x(RQ)
    qc.h(RQ)
    return qc

# --- Step 7: Grover circuit ---
iters = math.floor(math.pi / 4 * math.sqrt(N / M))
grover = QuantumCircuit(8)
# load the START board into the board register
start_no = BOARDS.index(START)
for q in range(5):
    if (start_no >> q) & 1:
        grover.x(BQ[q])
# superposition of all 8 rule sequences
grover.h(RQ)
for _ in range(iters):
    grover.compose(oracle(), inplace=True)
    grover.compose(diffuser(), inplace=True)
print("Grover iterations:", iters, " (floor(pi/4*sqrt(N/M)))")

# Exact success probability from the state vector, taken on the rule qubits only.
probs = Statevector(grover).probabilities(RQ)
exact = 0
for k in marked:
    exact = exact + probs[k]
print("Exact success probability:", round(exact, 4))

# --- Step 8: Measure the rule qubits on the simulator ---
qc = QuantumCircuit(8, STEPS)
qc.compose(grover, inplace=True)
qc.measure(RQ, range(STEPS))
sim = AerSimulator(seed_simulator=42)
result = sim.run(transpile(qc, sim), shots=1024).result()
counts = result.get_counts()
named = {}
for bits, c in counts.items():
    named[name(int(bits, 2))] = c
# sort the outcomes by shots, biggest first, and keep the top 5
ranked = sorted(named.items(), key=lambda item: -item[1])
top = dict(ranked[:5])
hits = 0
for seq, c in named.items():
    if replay(seq) == GOAL:
        hits = hits + c
print("\nTop outcomes (rule sequence -> shots):", top)
print("Measured success:", hits, "/ 1024 =", round(hits / 1024, 4))
# Draw on our own axes (ax=) so the chart also shows inside Jupyter.
fig, ax = plt.subplots(figsize=(7, 4))
plot_histogram(top, title="Grover over rule sequences (top 5 outcomes)", ax=ax)
plt.show()

# --- Step 9: Replay the winning sequence on the board (classical check) ---
seq = next(iter(top))
board = START
print("\nWinning sequence:", " ".join(seq))
for i, rule in enumerate(seq, 1):
    print("Step", i, "- rule", rule, "before:")
    show(board)
    board = apply_rule(board, rule)
    print("Step", i, "- rule", rule, "after:")
    show(board)
print("Reaches goal:", board == GOAL)
assert board == GOAL, "Grover result did not solve the puzzle"
avg_checks = (N + 1) / 2
print("\nClassical search: about", avg_checks,
      "checks on average for one marked sequence, Grover:", iters,
      "iterations (N is far too small for a real speed-up)")






























'''
================================================================================
VIVA AND THEORY NOTES - EXPERIMENT 5 : TARRATACA'S QUANTUM PRODUCTION SYSTEM
WITH THE 3-PUZZLE (GROVER OVER PRODUCTION-RULE SEQUENCES)
================================================================================

CODE EXPLANATION
----------------
1. Imports: math (pi, sqrt, floor), itertools (list all 24 boards), numpy (the rule matrix),
   matplotlib, qiskit QuantumCircuit + transpile, UnitaryGate (a gate given as a matrix),
   AerSimulator (local simulator), Statevector (exact probabilities), plot_histogram.
2. The puzzle: the 3-puzzle is a 2x2 board with tiles 1, 2, 3 and a blank (0). A board is a
   tuple of 4 numbers read row by row. GOAL = (1,2,3,0) and START = (2,0,1,3) are DEMO INPUT.
   BOARDS lists all 24 arrangements; the position of a board in this list is its board number
   (0..23), which fits in 5 qubits. show() prints a board as two text lines. apply_rule() is
   the production rule: V swaps the blank with its vertical neighbour, H swaps it with its
   horizontal neighbour. On a 2x2 board the blank always has one of each, so a rule is
   ALWAYS legal.
3. Rules as unitary operators: a 6-qubit gate (1 rule qubit + 5 board qubits) is written as a
   64 x 64 matrix U. It maps |rule>|board> to |rule>|board after the rule>. Each column has
   exactly one 1, so U is a permutation matrix and therefore unitary. The assert checks
   U @ U = I: a rule undoes itself, so applying the same gate again is the uncompute step.
   Board numbers 24..31 do not exist and are left unchanged. UnitaryGate turns U into a gate.
   (Building this matrix is classical preparation: the rule table written as a matrix.)
4. Counting M: the code tries all 8 rule sequences classically with replay() ONLY to count
   how many reach the goal (M = 1). Grover needs M to choose the number of iterations. The
   oracle does not contain the answers. The assert M > 0 stops the script if no sequence of
   3 rules reaches the goal.
5. Registers: 8 qubits. q0, q1, q2 are the rule qubits (rule of step 1, 2, 3: 0 = V, 1 = H).
   q3..q7 are the board register (board number in binary).
6. Oracle: (a) apply the rule gate of each step to the board register, controlled by that
   step's rule qubit; (b) goal test: X gates on the zero bits of the goal number, then a
   multi-controlled Z (H - multi-controlled X - H) gives a minus sign when the board equals
   GOAL; (c) apply the rule gates again in reverse order, which puts the board register back
   to START (uncomputation). The oracle only flips the phase of rule sequences that really
   lead to the goal; the rules are run, not looked up. The text drawing of the oracle is
   printed.
7. Diffuser: H, X on the rule qubits, multi-controlled Z, X, H. It reflects all amplitudes
   about their average (inversion about the mean).
8. Grover circuit: X gates load the START board number into the board register, H on the 3
   rule qubits makes an equal superposition of the 8 sequences, then oracle + diffuser are
   repeated floor(pi/4 * sqrt(N/M)) = 2 times.
9. Exact probability: Statevector(grover).probabilities(RQ) is the probability of each rule
   sequence (board register summed out); adding the marked sequence gives 0.9453.
10. Measurement: only the 3 rule qubits are measured, AerSimulator(seed_simulator=42),
   1024 shots. The bit string is turned back into a rule sequence (qubit 0 = step 1).
11. Histogram: top 5 outcomes drawn with plot_histogram on our own axes (ax=ax) so it also
   shows inside Jupyter.
12. Replay: the most frequent sequence is applied classically to the START board, printing
   the board before and after every step; the assert confirms the goal is reached.

EXPECTED OUTPUT (real run, local AerSimulator, seed 42)
----------------------------------------------------------------------
   Start board (demo input):  2 _ / 1 3          (printed as two lines)
   Goal board (demo input):   1 2 / 3 _
   Sequences N = 8  marked M = 1
   Oracle circuit (rules, goal test, undo):  (text drawing of 8 qubits: six "Rule" boxes,
      the goal test in the middle, three "Rule" boxes again at the end)
   Grover iterations: 2  (floor(pi/4*sqrt(N/M)))
   Exact success probability: 0.9453
   Top outcomes (rule sequence -> shots): {'HVH': 986, 'VHH': 7, 'HHH': 7, 'VHV': 6, 'HHV': 6}
   Measured success: 986 / 1024 = 0.9629
   Winning sequence: H V H
   Step 1 - rule H:  2 _ / 1 3  ->  _ 2 / 1 3
   Step 2 - rule V:  _ 2 / 1 3  ->  1 2 / _ 3
   Step 3 - rule H:  1 2 / _ 3  ->  1 2 / 3 _
   Reaches goal: True
   Classical search: about 4.5 checks on average for one marked sequence, Grover: 2
   iterations (N is far too small for a real speed-up)
   (The real script prints each board on two lines. One histogram is shown. The other 38
   shots are spread over the 7 wrong sequences. With another seed the stray counts change,
   the winner does not. Exact 0.9453 versus measured 0.9629 differ only by shot noise.)

OUTPUT GRAPHS EXPLAINED (and circuit diagrams for quantum scripts)
------------------------------------------------------------------
Each item: what it shows, how to read it, our result, a viva answer, a common mistake.
A. CIRCUIT DIAGRAM (text drawing, the oracle circuit)
   - Plots: the oracle on 8 qubits. q_0 to q_2 are rule qubits (step 1, 2, 3; 0 = V, 1 = H).
   - q_3 to q_7 are the 5 board qubits that hold the board number.
   - "Rule" box = a 64 x 64 gate; its small numbers 0 to 5 are inputs (wire 0 = rule qubit).
   - [ X ] = NOT; [ H ] = Hadamard; dots on q_3 to q_6 plus [ X ] on q_7 between H = multi-Z.
   - Read it: three Rule boxes play the moves; the middle flips the sign if board = goal;
   - then three Rule boxes in reverse undo the moves, so the board register returns to START.
   - Ours: 8 qubits, six Rule boxes, N = 8 sequences with M = 1 marked.
   - If asked: "The oracle plays the moves, marks the goal, then undoes the moves."
   - Misreading: the oracle does not measure or print the answer, it only flips a sign.

B. FIGURE 1 - BAR CHART "Grover over rule sequences (top 5 outcomes)"
   - Plots: shots per rule sequence (top 5 of 8). H = blank moves sideways, V = up or down.
   - x axis: rule sequence (HHH, HHV, HVH, VHH, VHV); y axis: Count (shots out of 1024).
   - Read it: one very tall bar with tiny others means the search worked.
   - With no search, all 8 sequences would get about 128 shots each.
   - Ours: HVH = 986 of 1024 = 0.9629 (exact 0.9453); other bars 7, 7, 6, 6. Replay hits goal.
   - If asked: "Grover lifted the winner from 1/8 to about 95 percent in 2 iterations."
   - Misreading: only 5 of 8 bars shown, so they do not sum to 1024; N = 8 gives no speed-up.

DETAILED THEORY NOTES
---------------------
1. WHAT IS A QUANTUM PRODUCTION SYSTEM?
   - A production system has three parts: STATES (descriptions of the problem), PRODUCTION
     RULES (actions that turn one state into another) and a GOAL test. Solving the problem
     means finding a sequence of rules from the start state to the goal. This is the classical
     AI idea behind puzzle solvers and expert systems.
   - Tarrataca's quantum production system does this on a quantum computer: the BOARD (state)
     is held in a quantum register, each production RULE is a unitary operator acting on that
     register, and the search over rule sequences is done with amplitude amplification
     (Grover style). The oracle runs the rules and tests the goal.
   - Type: quantum search / quantum AI. Not machine learning; nothing is trained.
   - Intuition: put a "rule register" in superposition of ALL rule sequences. For every
     sequence the rules are applied to the board register at the same time, the goal test marks
     (minus sign) the sequences that end in the goal, the rules are undone, and interference
     then makes the marked sequence very likely when we measure.
   - Maths in words: with N candidates and M solutions, Grover rotates the state by an angle
     2*theta per iteration, where sin(theta) = sqrt(M/N). Success probability after k
     iterations is sin^2((2k+1)*theta). Best k is about pi/4 * sqrt(N/M).
   - Our case: N = 8, M = 1, theta = asin(sqrt(1/8)) = 20.7 degrees. After 2 iterations
     sin^2(5 * 20.7 deg) = 0.9453, equal to the exact value printed by the code.

2. TYPES AND VARIANTS
   - Grover search (used here): unstructured search, quadratic speed-up, about sqrt(N) calls.
   - Amplitude amplification: generalised Grover that starts from any state, not only the
     uniform one; this is the idea behind the quantum production system.
   - Quantum counting: finds the number of solutions M (needed for the iteration count).
   - Quantum walks: search on graphs, a natural fit for the state graph of a puzzle.
   - QAOA / VQE (the manual mentions QAOA): variational heuristics that need a cost function
     and a classical optimiser; no guarantee of the optimum.
   - Classical search: BFS, DFS, A*, IDA*. BFS gives the shortest path; A* uses a heuristic.

3. KEY CONCEPTS AND TERMS
   - State / board : one arrangement of the tiles, for example (2,0,1,3). Here the board
     number (0..23) is stored in 5 qubits.
   - Production rule : here V (swap the blank with its vertical neighbour) and H (horizontal).
     Each rule is a permutation of the 24 boards, so it is a unitary operator.
   - Rule register : 3 qubits, one per step; the qubit value chooses V (0) or H (1).
   - Board register : 5 qubits holding the current board of the system.
   - Unitary operator : a reversible, probability-preserving transformation (U times its
     conjugate transpose is the identity). Permutation matrices are unitary.
   - Uncomputation : running the rules backwards so the board register returns to START and
     is not left entangled with the rule register. Without it, amplification would not work.
   - Goal test : checks if the board equals (1,2,3,0) and flips the phase if yes.
   - Search space : all rule sequences of the chosen length (here 2^3 = 8).
   - Superposition : (1/sqrt(8)) times the sum over all 8 sequences, equal amplitudes.
   - Oracle : circuit that flips the sign of sequences whose rules lead to the goal.
   - Diffuser : inversion about the mean; turns the sign difference into a larger amplitude.
   - Optimal iterations : floor(pi/4 * sqrt(N/M)). Too many iterations overshoots and the
     probability falls again.
   - Qiskit bit order : little-endian. Qubit 0 is the rightmost character of a printed count
     string. The script converts the number to a rule name with qubit 0 = step 1.
   - Shots : number of repeated runs; each shot gives one measured rule sequence.

4. HOW TO JUDGE THE RESULT
   - Success probability: exact 0.9453 (Statevector) and measured 986 / 1024 = 0.9629. Both
     close to 1 means Grover worked. They differ only by random shot noise.
   - Classical replay check: the measured sequence is applied to the board with apply_rule();
     the final board must equal the goal. This proves the answer is correct, not only likely.
   - Query count: classical search needs about (N+1)/2 = 4.5 checks on average for one
     marked item; Grover needs 2 iterations. With N = 8 this is far too small to call a real
     advantage; the point of the experiment is the construction.
   - Histogram: one tall bar (HVH) and small bars for the 7 wrong sequences.

5. SETUP (instead of a data split)
   - No dataset, no train/test split. The input is the START and GOAL board (demo input).
     The sequence length (3 rules) must be able to reach the goal; with no marked sequence
     (M = 0) the script stops at the assert. From this START, 2 or 4 rules cannot reach the
     goal (M = 0); odd lengths such as 3 or 5 can.
   - Only 12 of the 24 boards can be reached from any given board (the other 12 are the
     "unsolvable" half of the 3-puzzle). Our GOAL is in the reachable half of START.
   - Seed: AerSimulator(seed_simulator=42) fixes the sampling so the counts repeat.

6. THE SIMULATOR (AerSimulator)
   - qiskit_aer.AerSimulator runs the circuit on the local computer. No IBM Quantum account,
     no API token, no internet. The manual's code uses Aer.get_backend('qasm_simulator') with
     execute() and qiskit.aqua (QAOA with COBYLA); these are removed, so AerSimulator with run()
     replaces them. For real hardware use qiskit-ibm-runtime (QiskitRuntimeService and a
     real backend, where the IBMQ line of the manual would go).
   - 8 qubits is tiny, so simulation is instant. A simulator stores all 2^n amplitudes (256
     here); this grows exponentially, which is why simulating many qubits becomes impossible.
   - WHAT IS QUANTUM HERE: the superposition over all 8 rule sequences; the rules acting as
     unitary operators on the board register, controlled by the rule qubits; the uncomputation
     step; amplitude amplification (oracle + diffuser).
   - WHAT IS CLASSICAL HERE: building the rule permutation matrices (a classical loop over
     all 24 boards), counting M by trying all 8 sequences (only to choose the iteration
     count), reading the exact probability, and the final replay check. The oracle itself
     does not contain the solutions, but the rule matrices were written by a classical program.
   - WHY THE RULES CAN BE UNITARY: on a 2x2 board every rule (V or H) is legal in every
     board, and each rule just permutes the boards, so it is reversible and a valid gate. An
     earlier U/D/L/R version had illegal moves (the blank cannot leave the board); "do nothing
     if illegal" merges different boards into one and is NOT reversible, so it cannot be a
     unitary gate. Choosing V and H avoids the problem.
   - Note: the manual describes the 3-puzzle as three tiles in a row moving left/right, while
     this script uses the standard 2x2 sliding 3-puzzle.
   - Note: a dense 64 x 64 matrix is fine for a demo; for large boards the rules would have
     to be built from small gates, otherwise the matrix grows exponentially.

7. ADVANTAGES
   - Quadratic speed-up over unstructured classical search (about sqrt(N) instead of N).
   - Searches all rule sequences in superposition at the same time.
   - The oracle really runs the rules, so no answer has to be known in advance (apart from
     the count M used for the iteration number).
   - Same method works for any production system (puzzles, planning, scheduling).
   - The answer is verified classically in one pass.

8. DISADVANTAGES
   - Only a quadratic gain, not exponential; classical heuristic search (A*) or BFS is often
     better on structured problems and wins easily on 8 sequences.
   - The oracle needs the rules and goal test as reversible circuits, with uncomputation,
     extra qubits (board register) and many gates.
   - Needs M (or quantum counting) to choose the number of iterations.
   - Search depth must be fixed in advance; the shortest solution needs runs with
     different depths.
   - Qubits grow with depth (one per step here); noisy hardware cannot run deep circuits.
   - Result is probabilistic, so it must be checked classically.

9. REAL-WORLD USES
   - Search and planning in AI (puzzles, route and schedule search).
   - Key search or database search (quadratic speed-up of brute force).
   - Constraint satisfaction and SAT style problems.
   - Subroutine inside other quantum algorithms (counting, minimum finding).

10. COMPARISON WITH RELATED METHODS
   - Quantum production system vs classical BFS: BFS finds the shortest path and uses the
     puzzle structure; the quantum system searches all sequences of a fixed length with about
     sqrt(N) oracle calls.
   - Grover vs QAOA: Grover is amplitude amplification with a proven quadratic gain; QAOA is
     a variational heuristic that needs a classical optimiser and a cost function.
   - Grover vs A*: A* uses a heuristic to prune the tree; Grover does not prune, it amplifies.
   - Grover vs Shor: Grover is a quadratic speed-up for search; Shor is an exponential
     speed-up for factoring.
   - This version vs a lookup-table oracle: a lookup oracle writes the classical answers
     into the circuit (cheating); this version computes the goal test with the rules.

LIBRARIES AND FUNCTIONS USED
----------------------------
- math : pi, sqrt, floor for the optimal iteration count.
- itertools.permutations(range(4)) : all 24 arrangements of the tiles 0..3 (the boards).
- numpy : np.zeros and np.eye build matrices, np.allclose checks U @ U equals the identity.
- matplotlib.pyplot (plt) : plt.subplots makes the figure and axes, plt.show displays it.
- qiskit.QuantumCircuit(n, name=...) : a circuit with n qubits (second argument = classical
  bits); methods h (Hadamard), x (NOT), mcx (multi-controlled X), append (add a gate on
  chosen qubits), compose (join two circuits), copy, measure, draw(output="text", fold=100).
- qiskit.circuit.library.UnitaryGate(matrix, label) : turns a unitary matrix into a gate;
  qiskit raises an error if the matrix is not unitary.
- qiskit.transpile : rewrites the circuit into gates the backend supports.
- qiskit_aer.AerSimulator(seed_simulator=42) : local simulator; run(circuit, shots=1024)
  then result().get_counts() returns a dictionary {bitstring: count}.
- qiskit.quantum_info.Statevector : Statevector(circuit).probabilities(qubits) gives exact
  probabilities of the listed qubits (other qubits are summed out).
- qiskit.visualization.plot_histogram(data, ax=ax, title=...) : bar chart of counts, drawn on
  the given axes so it shows in Jupyter.
- own functions : show (print a board), apply_rule (one production rule), name (number to
  rule sequence), replay (apply a sequence), oracle, diffuser.
- Python: i ^ 2 and i ^ 1 (flip a bit = vertical or horizontal neighbour on the 2x2 board),
  x >> 1 and x & 1 (split the matrix index into board and rule), next(iter(dict)) (first
  key), sorted with a key (top outcomes), int(bits, 2) (binary string to number).

DETAILED VIVA Q&A
-----------------
Q1. How does Tarrataca's quantum production system encode the problem state as a quantum state?
A1. Each state of the problem (a board arrangement) is written as a number in binary and
    stored in a register of qubits, one basis state per board. In our script the 24 boards of
    the 2x2 puzzle get board numbers 0..23, which fit in a 5-qubit board register. Several
    boards can be held at once as a superposition of basis states. A second register (3 rule
    qubits) holds the sequence of rules in superposition.

Q2. How does Tarrataca's quantum production system construct a quantum circuit that represents the problem space?
A2. The production rules (moves) are written as unitary operators, because every quantum
    gate must be reversible. Each rule is a permutation of the boards, so it is a permutation
    matrix. The rule gate is controlled by a rule qubit and acts on the board register. Applying
    the rule gates for step 1, 2, 3 in turn builds, in superposition, the boards reached by
    all 8 rule sequences. A goal test then flips the phase when the board equals the goal, and
    the rules are undone. Oracle plus diffuser form the Grover iteration.

Q3. What are some of the quantum algorithms that can be used to search for the optimal solution to the 3-puzzle problem using Tarrataca's quantum production system?
A3. Grover's search / amplitude amplification (used here), quantum counting to find the
    number of solutions, quantum walks on the state graph, and variational methods such as
    QAOA (mentioned in the manual). Grover and amplitude amplification are the ones with a
    proven speed-up.

Q4. How can we measure the qubits at the end of the quantum circuit to obtain the optimal solution to the 3-puzzle problem?
A4. We measure the rule qubits in the computational basis and repeat for many shots. Each
    shot gives a bit string, which is read as a rule sequence (qubit 0 = step 1). The most
    frequent bit string is the answer (HVH, 986 of 1024 shots). The board register is not
    measured; it was uncomputed. Then we replay the sequence classically to confirm the goal.

Q5. What are some of the advantages and disadvantages of using Tarrataca's quantum production system to solve the 3-puzzle problem?
A5. Advantages: quadratic speed-up in the number of oracle calls, searches all sequences in
    superposition, general for any production system, the oracle really runs the rules.
    Disadvantages: only a quadratic gain (classical BFS or A* is cheaper on such a tiny puzzle),
    the rules and goal test must be built as reversible circuits with uncomputation and extra
    qubits, the depth must be fixed in advance, and noisy hardware cannot run big instances.

Q6. What is a production system?
A6. It is made of states, production rules (actions that change a state) and a goal. Solving
    it means finding a chain of rules from the start state to the goal state.

Q7. What is the 3-puzzle?
A7. A 2x2 board with tiles 1, 2, 3 and one blank. A move slides a tile into the blank. The
    goal here is 1 2 / 3 blank.

Q8. How many qubits do we use and why?
A8. 8 qubits: 3 rule qubits (one per step, because each step has only 2 rules, V or H) plus
    5 board qubits (24 boards need 5 bits since 2^5 = 32 >= 24). The sequence space is
    2^3 = 8 sequences.

Q9. What is Grover's algorithm?
A9. A quantum search algorithm. It starts with all candidates in equal superposition and
    repeats oracle (marks the solution with a minus sign) and diffuser (inversion about the
    average), so the solution probability grows. It needs about sqrt(N) steps instead of N.

Q10. What does the oracle do in this code?
A10. It applies the chosen rules to the START board, flips the sign if the board equals the
     GOAL, and then undoes the rules. Only sequences that really lead to the goal get the
     minus sign. It contains the rules and the goal test, not the list of answers.

Q11. What does the diffuser do?
A11. It reflects every amplitude about the average amplitude. Because the oracle made the
     solution amplitude negative (below average), the reflection makes it much larger and the
     others smaller.

Q12. Why is uncomputation needed?
A12. After the rules run, the board register holds a board that depends on the rule
     sequence, so the two registers are entangled. If we left it, the diffuser on the rule
     qubits could not interfere the sequences properly. Running the rules backwards returns
     the board register to START for every sequence, so only the phase is kept.

Q13. How many Grover iterations are used and why?
A13. floor(pi/4 * sqrt(N/M)) = floor(pi/4 * sqrt(8)) = 2. After 2 the success probability is
     0.9453. One iteration gives about 0.78 and three give about 0.33 (overshoot).

Q14. Where does M come from? Is that cheating?
A14. We count M by trying all 8 sequences classically, but only to choose the number of
     iterations. Grover needs M for that. The oracle does not contain the answers. In a real
     use M can come from quantum counting or be guessed by trying iteration counts 1, 2, 4, ...

Q15. What is the success probability and how is it obtained?
A15. Exact: 0.9453 from Statevector probabilities on the rule qubits. Measured: 986 of 1024
     shots = 0.9629. The small difference is random shot sampling.

Q16. Why must the rules be unitary?
A16. Quantum gates must be reversible and preserve probability. A permutation of the boards
     is reversible, so its permutation matrix is unitary and can be a gate.

Q17. Why did the earlier U/D/L/R version not work as a unitary?
A17. On a 2x2 board a move can be illegal (the blank would leave the board). "Do nothing when
     illegal" sends two different boards to the same board, which is not reversible, so it is
     not a permutation and not unitary. V and H are always legal on a 2x2 board, so they are
     permutations.

Q18. What do the rules V and H do?
A18. V swaps the blank with the tile above or below it, H swaps the blank with the tile left
     or right of it. Applying the same rule twice undoes it, so each rule is its own inverse.

Q19. Is every board reachable from every other board?
A19. No. Only half of the 24 boards (12) can be reached from any given board; the other 12 are
     in the unsolvable half. Our START and GOAL are in the same half.

Q20. What if there are several solutions?
A20. M becomes larger, the iteration count drops to floor(pi/4 * sqrt(N/M)), and the
     measurement returns any one of the solutions.

Q21. What if there is no solution (M = 0)?
A21. The assert M > 0 stops the script with a message before Grover runs. In theory Grover
     with nothing marked just stays a uniform random guess.

Q22. What is the qubit order in the printed bit string?
A22. Qiskit is little-endian: qubit 0 is the rightmost bit. The script turns each count key
     into a rule sequence with qubit 0 as step 1, so the name reads in time order.

Q23. What is the difference between Grover search and QAOA?
A23. Grover has a proven quadratic speed-up and a fixed recipe. QAOA is variational: circuit
     parameters are tuned by a classical optimiser to lower a cost, giving approximate answers
     with no guarantee.

Q24. What is amplitude amplification?
A24. A generalisation of Grover's search: any quantum procedure that finds the answer with
     probability p can be boosted to nearly 1 using about 1/sqrt(p) repetitions.

Q25. Is there a real speed-up in this demo?
A25. No. With only 8 sequences, a classical search needs about 4.5 checks on average and
     Grover uses 2 iterations, but each iteration runs the whole rule circuit and the
     simulator costs more than the classical search. The point is the construction, not the
     speed. A real advantage would need a big search space.

Q26. Why do we replay the answer classically?
A26. Grover is probabilistic. Replaying the sequence with apply_rule() proves it really solves
     the puzzle, and checking one sequence is cheap.

Q27. Can this method guarantee the shortest solution?
A27. Not by itself. We fix the length (3). To get the shortest we run depth 1, 2, 3, ... and take
     the first depth that has a solution (as BFS does).

Q28. What is quantum and what is classical in this script?
A28. Quantum: superposition of rule sequences, rules as unitary operators on the board
     register, uncomputation, amplitude amplification. Classical: building the permutation
     matrices, counting M, reading the exact probability, and the final replay check.

Q29. How does a real quantum computer differ from this simulator?
A29. A real device has noise and gate errors and returns slightly different counts every run.
     The AerSimulator without a noise model is ideal; the only randomness is shot sampling,
     which the seed makes repeatable.

Q30. How does the manual's 3-puzzle differ from this script?
A30. The manual describes three tiles in a row moving left/right. This script uses the
     standard 2x2 sliding 3-puzzle, which has the same idea (states, rules, goal) but a
     small board where every rule is always legal.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Change START to (1,2,0,3). What happens?
A1. That board is 1 rule (H) from the goal. The 3-rule sequences that reach the goal are VVH,
    HVV and HHH, so M = 3 and the iteration count becomes floor(pi/4 * sqrt(8/3)) = 1.
    Any of the three can be measured; the success probability after 1 iteration is about
    0.84.

Q2. Change STEPS to 4. What happens?
A2. With this START, 4 rules can never reach the goal (every rule changes the parity of the
    board and the goal is an odd number of steps away), so M = 0 and the assert stops the
    script. Use 5 instead: N = 32, 5 rule qubits (10 qubits in total), M = 5. Note RQ
    must also be changed to list the new rule qubits and BQ to follow them.

Q3. Write the line that computes the number of Grover iterations.
A3. iters = math.floor(math.pi / 4 * math.sqrt(N / M))

Q4. Which line makes the equal superposition of rule sequences?
A4. grover.h(RQ) puts a Hadamard on each rule qubit, so all 8 sequences have amplitude
    1/sqrt(8).

Q5. Set iters = 1 or iters = 3. What happens to the success probability?
A5. With 1 iteration it is about 0.78 (sin^2(3*20.7 deg)). With 3 it rotates past the target
    and falls to about 0.33 (sin^2(7*20.7 deg)). So the right count matters.

Q6. Write the line that runs the circuit on the simulator for 1024 shots.
A6. result = sim.run(transpile(qc, sim), shots=1024).result() then counts = result.get_counts()

Q7. How do you change the seed and what do you expect?
A7. AerSimulator(seed_simulator=7). The top outcome stays HVH; only the few stray shots and
    exact counts change.

Q8. How is the multi-controlled Z written in the code?
A8. qc.h(BQ[4]), qc.mcx(BQ[:4], BQ[4]), qc.h(BQ[4]). H turns the target X into a Z. The X gates
    before and after it select the board number of the goal.

Q9. What would happen if you remove the second loop of rule gates in the oracle (no
    uncomputation)?
A9. The board register would stay entangled with the rule register, so the diffuser would not
    amplify the marked sequence correctly and the success probability would drop a lot.

Q10. How would you add a noise model?
A10. Build a qiskit_aer.noise.NoiseModel, add depolarising errors to gates, and pass it to
     AerSimulator(noise_model=...). The success probability drops because the circuit has
     many gates (each Rule gate is large once transpiled).

Q11. How can you print the circuit?
A11. print(grover.draw(output="text", fold=100)) or print(oracle().draw(output="text")). Use
     grover.decompose() to see inside the oracle and diffuser blocks.

Q12. How would you test a different goal?
A12. Change GOAL, for example (1,2,3,0) to (1,0,3,2). Choose a START that is in the same
     reachable half and an odd or even distance matching STEPS; the classical count recomputes
     M and the oracle follows because the goal test uses BOARDS.index(GOAL). Only 12 of the 24
     boards can be reached from a given board.

Q13. Write the line that checks that the rule matrix is unitary.
A13. assert np.allclose(U @ U.T, np.eye(64)) (for a real permutation matrix U.T is the inverse
     U^dagger). The script checks U @ U = I, which also shows each rule is its own inverse.

END OF VIVA NOTES
'''
