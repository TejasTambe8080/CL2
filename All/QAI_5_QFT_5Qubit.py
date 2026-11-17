"""
Practical 3 : Quantum Fourier Transform (5 qubits)
==================================================
INSTRUCTIONS
 1. Install:  pip install qiskit qiskit-aer numpy matplotlib
 2. Dataset : None needed. Everything runs on qiskit's local AerSimulator
              (no IBM account/token).
 3. Run     : python QAI_5_QFT_5Qubit.py
 4. Steps   : encode a value with X gates -> QFT -> measure (spread out) ->
              QFT then inverse QFT (value comes back) -> build QFT by hand
              (H, cp, swap) -> compare with library QFT
"""

# --- Step 1: Import the libraries ---
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram

n = 5
sim = AerSimulator(seed_simulator=42)

# --- Step 2: Encode a value with X gates ---
# X on q0, q2 and q4 gives binary 10101 = 21.
def encode(qc):
    qc.x([0, 2, 4])

# --- Step 3: Apply QFT and measure with 1000 shots ---
qc = QuantumCircuit(n, n)
encode(qc)
qc.append(QFTGate(n), range(n))
qc.measure(range(n), range(n))
job = sim.run(transpile(qc, sim), shots=1000)
counts_qft = job.result().get_counts()
print("QFT only: number of different outcomes =", len(counts_qft), "out of", 2 ** n)
ranked = sorted(counts_qft.items(), key=lambda kv: -kv[1])
print("Five most common:", ranked[:5])

# --- Step 4: Apply QFT and then inverse QFT; the encoded value must come back ---
qc2 = QuantumCircuit(n, n)
encode(qc2)
qc2.append(QFTGate(n), range(n))
qc2.append(QFTGate(n).inverse(), range(n))
qc2.measure(range(n), range(n))
job2 = sim.run(transpile(qc2, sim), shots=1000)
counts_back = job2.result().get_counts()
print("QFT then inverse QFT:", counts_back)

# --- Step 5: Draw the QFT circuit as text ---
# decompose shows the H, cp and swap gates inside.
base = QuantumCircuit(n)
qft_circuit = base.compose(QFTGate(n), range(n))
print(qft_circuit.decompose().draw(output="text", fold=120))

# --- Step 6: Build the 5-qubit QFT by hand ---
# H, then controlled-phase (cp) gates, then swaps.
hand = QuantumCircuit(n)
for j in range(n - 1, -1, -1):
    hand.h(j)
    for k in range(j - 1, -1, -1):
        hand.cp(np.pi / 2 ** (j - k), k, j)     # rotation pi/2, pi/4, pi/8, pi/16
for i in range(n // 2):
    hand.swap(i, n - 1 - i)                     # reverse the qubit order
print(hand.draw(output="text", fold=120))

# --- Step 7: Check that the hand-made circuit equals the library QFT ---
same = np.allclose(Operator(hand).data, Operator(QFTGate(n)).data)
print("Hand-made QFT equals library QFT:", same)

# --- Step 8: Plot the two histograms ---
# Draw on our own axes (ax=) so the chart also shows inside Jupyter.
fig, ax = plt.subplots(figsize=(12, 5))
labels = ["QFT only", "QFT then inverse QFT"]
plot_histogram([counts_qft, counts_back], legend=labels, ax=ax)
plt.show()






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 3 : QUANTUM FOURIER TRANSFORM (5 QUBITS)
================================================================================

CODE EXPLANATION
----------------
1. Imports: numpy (pi and allclose), matplotlib, QuantumCircuit and transpile (qiskit),
   AerSimulator (qiskit_aer), QFTGate (qiskit.circuit.library), Operator (qiskit.quantum_info)
   and plot_histogram. In qiskit 2.x the old QFT circuit class is deprecated; QFTGate is the
   current form. The manual used execute() and IBMQ, which are removed, so AerSimulator is used
   locally (no IBM account or token). Real hardware would need qiskit-ibm-runtime.
2. n = 5 qubits. sim = AerSimulator(seed_simulator=42) so the counts repeat.
3. encode(qc): X gates on q0, q2, q4 as in the manual. Qiskit writes qubit 4 on the left, so
   the input is the bitstring 10101 (decimal 21). (The manual quotes 1010 for its 4-bit view.)
4. QFT only: encode, append QFTGate(5) on all qubits, measure all, run 1000 shots. The QFT turns
   a basis state into an equal-size mix of all 32 states with different phases, so the
   measurement is spread almost evenly over all 32 outcomes.
5. QFT then inverse QFT: QFTGate(5).inverse() undoes the QFT exactly, so the input 10101 comes
   back in all 1000 shots.
6. Circuit drawing: compose QFTGate into a plain circuit and decompose() it once so the H, P
   (controlled phase) and swap gates inside are visible.
7. Hand-made QFT: for j from the top qubit down to 0: H on j, then controlled-phase gates
   cp(pi / 2^(j-k)) from every lower qubit k to j. That gives the rotation angles pi/2, pi/4,
   pi/8, pi/16. After that swap qubit i with qubit n-1-i to reverse the order.
   The manual used cu1(theta, ...); cu1 is now called cp (controlled phase).
8. Operator(hand).data is the 32 x 32 unitary matrix of the circuit. np.allclose compares it
   with the library QFT matrix. True means the two circuits are identical.
9. plot_histogram draws both result sets.

EXPECTED OUTPUT (real run, AerSimulator, seed_simulator=42)
----------------------------------------------------------------------
   QFT only: number of different outcomes = 32 out of 32
   Five most common: [('01110', 43), ('01100', 40), ('00101', 40), ('11000', 39), ('10100', 36)]
   QFT then inverse QFT: {'10101': 1000}
   (two circuit drawings: library QFT decomposed and the hand-made QFT; both show
    H, P(pi/2), P(pi/4), P(pi/8), P(pi/16) and swaps)
   Hand-made QFT equals library QFT: True
   (An ideal uniform spread would give 1000/32 = about 31 per outcome; we see about 20 to 43,
   which is normal random variation. The exact counts can differ on another machine, but all
   32 outcomes appear, and the second test always gives 10101 with 1000 of 1000 shots.)

OUTPUT GRAPHS EXPLAINED (and circuit diagrams)
----------------------------------------------
Three outputs: two circuit drawings and one histogram.

A. CIRCUIT DIAGRAM 1 - library QFT, decomposed (first drawing)
   - What it shows: QFTGate(5) opened into basic gates; time runs from left to right.
   - The wide drawing is folded; a double-arrow mark means it continues in the next lines.
   - Wires q_0 to q_4. [H] = Hadamard; two [X] joined by a line = SWAP.
   - A dot joined to a box P(pi/2) = controlled phase: turns the target if the control is 1.
   - Read: H on q_4, then P gates from q_3 down to q_0, then H on q_3, and so on; swaps last.
   - Our result: H, P(pi/2), P(pi/4), P(pi/8), P(pi/16) and swaps.
   - If asked: "QFT = H on each qubit plus phase turns that halve each time, then swaps."
   - Common misreading: P(pi/2) is the target half of a controlled gate, not a lone gate.

B. CIRCUIT DIAGRAM 2 - hand-made QFT (second drawing)
   - What it shows: the same QFT built by our own loop of h, cp and swap.
   - Symbols: the same as diagram 1 (H, dot + P box, joined crosses for SWAP).
   - How to read: compare it with diagram 1 block by block; the gate pattern is the same.
   - Each angle is pi divided by 2 to the power of the gap between the two qubits.
   - Operator(hand).data and np.allclose compare the two 32 x 32 matrices.
   - Our result: Hand-made QFT equals library QFT: True.
   - If asked: "I built QFT from H, cp and swap, and np.allclose proved it equals the library."
   - Common misreading: looking alike is not the proof; the matrix check is.

C. FIGURE 1 - histogram (blue = QFT only, orange = QFT then inverse QFT)
   - What it plots: how often each 5-bit result came up in 1000 shots, for two runs.
   - X axis: the measured 5-bit string (32 labels). Y axis: Count of shots.
   - The number above each bar is its exact count.
   - Many short bars of similar height = even spread; one tall bar = one sure result.
   - Blue is flat as QFT spreads a basis state; orange is one bar as the inverse undoes it.
   - Our result: QFT only gives 32 of 32 outcomes; QFT then inverse QFT gives {'10101': 1000}.
   - If asked: "QFT alone gives a flat spread; QFT then inverse QFT returns 10101 for sure."
   - Common misreading: the uneven blue heights (about 20 to 43) are shot noise, not an error.

DETAILED THEORY NOTES
---------------------
1. WHAT IS THE QUANTUM FOURIER TRANSFORM (QFT)?
   - Definition: the QFT is the quantum version of the discrete Fourier transform (DFT). It is a
     unitary operation that maps the amplitudes of a quantum state to their Fourier
     coefficients.
   - Type: quantum circuit / subroutine. It is a building block, not a full algorithm.
   - Intuition: the classical DFT finds which frequencies are present in a signal. The QFT does
     the same for a signal stored in the amplitudes of qubits. A basis state |x> (a single
     "value") becomes a wave with a frequency set by x, so every output is equally likely but
     each has a different phase. The information is now stored in the phases.
   - Maths in words: for N = 2^n, QFT|x> = (1/sqrt(N)) * sum over y of exp(2*pi*i*x*y/N) |y>.
     So each basis state spreads into all N basis states with the same size (1/sqrt(N)) and a
     phase that depends on x*y. Because the sizes are equal, measuring gives a nearly flat
     distribution for any input basis state.
   - Circuit: for each qubit apply H, then controlled phase rotations of pi/2, pi/4, pi/8, ...
     from the less significant qubits, and finally swap the qubits to reverse their order.
   - Number of gates: n Hadamards + n(n-1)/2 controlled phases + about n/2 swaps = O(n^2).

2. TYPES AND VARIANTS
   - Exact QFT (this practical) : all rotations used, O(n^2) gates.
   - Approximate QFT (AQFT) : leaves out the tiny rotations (like pi/2^10) to cut gates to
     O(n log n) with very small error. Used on real hardware because small rotations are too
     noisy.
   - Inverse QFT : the reverse circuit (QFTGate(n).inverse()). Used at the end of phase
     estimation.
   - QFT without final swaps : swaps are often dropped and qubit order is read in reverse
     (do_swaps option / classical relabelling).
   - Classical relatives : DFT (O(N^2)), FFT (O(N log N)), inverse DFT.
   - Qiskit : QFTGate (current) and QFT (older circuit class, deprecated in 2.x).

3. KEY CONCEPTS AND TERMS
   - Unitary : a reversible operation that keeps probabilities adding to 1; every quantum gate
     is unitary.
   - DFT : discrete Fourier transform of N numbers: X[k] = sum over n of x[n] exp(-2*pi*i*k*n/N).
   - Phase : the angle of a complex amplitude; changing it does not change the probability of
     one state but changes how states interfere.
   - Hadamard (H) : creates superposition; it is the 1-qubit QFT.
   - Controlled phase gate (cp(theta, control, target)) : applies a phase exp(i*theta) to the
     target only if the control is 1. Older name: cu1.
   - Swap gate : exchanges two qubits. The QFT circuit produces the output in reverse bit
     order, so the swaps put it right.
   - Operator : qiskit.quantum_info.Operator gives the full unitary matrix of a circuit.
   - np.allclose : True if two arrays are equal within a small tolerance (needed because of
     floating point errors).
   - Basis state / computational basis : |00000> to |11111>, i.e. numbers 0 to 31.
   - Phase estimation : an algorithm that finds the eigenvalue phase of a unitary; it uses the
     inverse QFT.
   - Little endian : Qiskit prints qubit 0 on the right of a bitstring.

4. HOW TO JUDGE THE RESULT
   - QFT only: expect ALL 32 outcomes with about 31 shots each (flat). We got 32 of 32 outcomes.
     Reason: a single basis state goes to an equal superposition with different phases.
   - QFT followed by inverse QFT: must return the input in 100 percent of shots. We got 10101
     in 1000 of 1000 shots.
   - Hand-made circuit check: np.allclose(Operator(hand).data, Operator(QFTGate(5)).data) must
     be True, meaning the two unitary matrices are identical.
   - Note that measuring right after a QFT of a basis state hides the information in phases;
     the flat result is expected, it does not mean the QFT failed.

5. DATA SPLIT / RELEVANT SETUP
   - No data and no train/test split. The settings are n = 5 qubits (so N = 32 values), the
     encoded value (10101), shots (1000) and the seed.

6. THE SIMULATOR (AerSimulator)
   - Exact state-vector style simulation of 5 qubits (32 amplitudes) on the classical
     computer; noise free unless a noise model is given.
   - seed_simulator=42 fixes the random choices in the measurement sampling.
   - No account or token is needed. transpile() converts QFTGate and the inverse into basic
     gates the simulator understands.

7. ADVANTAGES
   - Needs only O(n^2) gates for 2^n amplitudes; the classical FFT needs O(n 2^n) operations.
   - It is the key part of Shor's factoring algorithm and quantum phase estimation.
   - Exactly reversible (inverse QFT is the same circuit backwards).
   - Can be approximated cheaply (AQFT) for hardware.

8. DISADVANTAGES
   - You cannot read all 2^n Fourier coefficients: a measurement gives only one sample. The QFT
     is useful only inside a larger algorithm that makes the answer appear as a peak.
   - Loading classical data into amplitudes is expensive and can cancel the speed-up.
   - Many small-angle controlled rotations are hard on noisy hardware.
   - Needs good control of all-to-all qubit connections (many two-qubit gates).

9. REAL-WORLD USES
   - Shor's algorithm (finding the period used for factoring large numbers).
   - Quantum phase estimation, used in quantum chemistry and in the HHL linear-system
     algorithm.
   - Hidden subgroup problems, period finding (Simon, Deutsch-Jozsa relatives).
   - Quantum arithmetic (QFT based adders and multipliers).
   - Quantum signal and data analysis ideas.

10. COMPARISON WITH RELATED METHODS
   - QFT vs classical DFT : same maths; DFT on N = 2^n points costs O(N^2), the QFT circuit
     uses O(n^2) gates, but the output sits in amplitudes and cannot be fully read.
   - QFT vs classical FFT : FFT needs O(n 2^n) = O(N log N) operations; the QFT needs O(n^2)
     gates, which is exponentially fewer, again with the readout limit.
   - QFT vs Hadamard transform : the Hadamard on every qubit is the Fourier transform over
     bits (used in Deutsch-Jozsa); the QFT is over the integers mod N.
   - Exact QFT vs approximate QFT : exact uses all rotations, approximate drops tiny ones for
     fewer gates and similar accuracy.

LIBRARIES AND FUNCTIONS USED
----------------------------
- numpy (np) : np.pi, np.allclose (compare matrices).
- matplotlib.pyplot (plt) : show (display the histogram).
- qiskit.QuantumCircuit(n, n) : circuit with n qubits and n classical bits; QuantumCircuit(n)
  without classical bits for the hand-made version.
- qc.x([0, 2, 4]) : X (NOT) gate on several qubits. qc.h(j) : Hadamard. qc.cp(theta, c, t) :
  controlled phase. qc.swap(a, b) : swap two qubits.
- qc.append(gate, qubits) : adds a gate (here QFTGate) on the listed qubits.
- qc.compose(other, qubits) : places another circuit onto the listed qubits.
- qc.decompose() : opens one level of a composite gate to show the gates inside.
- qc.measure(range(n), range(n)) : measure qubit i into bit i.
- qc.draw(output="text", fold=120) : ASCII drawing, wrapped at 120 characters.
- qiskit.circuit.library.QFTGate(n) : the n-qubit QFT as a gate; .inverse() gives the inverse.
- qiskit.quantum_info.Operator(circuit).data : the unitary matrix of the circuit.
- qiskit.transpile, qiskit_aer.AerSimulator(seed_simulator=42), sim.run(...).result()
  .get_counts() : run and read counts.
- qiskit.visualization.plot_histogram(list, legend=[...], figsize=...) : bar chart of counts.
- sorted(counts.items(), key=lambda kv: -kv[1])[:5] : the five most frequent outcomes.

DETAILED VIVA Q&A
-----------------
Q1. What is the Quantum Fourier Transform?
A1. It is the quantum circuit that performs the discrete Fourier transform on the amplitudes of
    n qubits. It maps |x> to (1/sqrt(N)) sum of exp(2*pi*i*x*y/N)|y> for N = 2^n, and is
    unitary, so it is reversible.

Q2. What is the difference between the QFT and the classical DFT?
A2. The DFT takes N numbers and returns N numbers (O(N^2) steps, or O(N log N) with the FFT).
    The QFT acts on the 2^n amplitudes of n qubits with only O(n^2) gates. But the result is
    stored in amplitudes and cannot all be read out; a measurement gives only one outcome.

Q3. Why are swap gates used at the end?
A3. The H and controlled phase gates produce the Fourier output with the qubit order reversed.
    Swapping qubit i with qubit n-1-i puts the bits in the normal order.

Q4. What is the complexity of the QFT compared with the FFT?
A4. QFT: O(n^2) gates for n qubits (N = 2^n values). FFT: O(n 2^n) operations. So the QFT is
    exponentially cheaper in gate count, but the output cannot be fully read out.

Q5. Where is the QFT used?
A5. Shor's factoring algorithm (period finding), quantum phase estimation, the HHL algorithm,
    quantum chemistry simulations and quantum arithmetic.

Q6. What gates are inside the QFT circuit?
A6. Hadamard gates, controlled phase gates with angles pi/2, pi/4, pi/8, pi/16 (for 5 qubits)
    and swap gates.

Q7. What is a controlled phase gate? What was cu1?
A7. cp(theta, control, target) multiplies the |11> component by exp(i*theta); i.e. it adds a
    phase to the target only if the control is 1. cu1 was the old name for the same gate. It
    was renamed cp.

Q8. Why do we get all 32 outcomes after QFT?
A8. A basis state like 10101 goes to a state where all 32 outcomes have the same size
    amplitude 1/sqrt(32) and different phases, so each has probability 1/32. A measurement
    shows them all about equally often.

Q9. Then how does the QFT carry information?
A9. In the phases. The inverse QFT can bring the phases back to a single basis state, which is
    why QFT followed by inverse QFT returns 10101 every time.

Q10. What happens when you apply QFT and then the inverse QFT?
A10. The two cancel (unitary times its inverse equals the identity). The input comes back with
     probability 1, which we saw as {'10101': 1000}.

Q11. What is the meaning of X gates on q0, q2 and q4?
A11. X flips a qubit from 0 to 1. Putting X on q0, q2 and q4 makes the input state |10101>,
     the binary number 21. It is just a way to put a known value into the register.

Q12. Why is the input called 1010 in the manual but 10101 here?
A12. The manual's text refers to a 4-bit pattern; with X on q0, q2, q4 of 5 qubits the input in
     Qiskit's notation (qubit 4 on the left) is 10101. The code prints what it really gets.

Q13. What are the rotation angles in a 5-qubit QFT?
A13. pi/2, pi/4, pi/8 and pi/16. In general the rotation between qubits j apart is
     pi / 2^j (which is 2*pi / 2^(j+1)).

Q14. How many gates does the 5-qubit QFT need?
A14. 5 Hadamards, 5*4/2 = 10 controlled phase gates and 2 swaps: 17 gates in total, which is
     about n^2 / 2.

Q15. What is QFTGate and why not QFT?
A15. QFTGate is the current way to get the QFT as a gate in qiskit 2.x. The older QFT class
     (a circuit) is deprecated. Both build the same unitary.

Q16. How do you verify the hand-made QFT?
A16. Compute the unitary matrices with Operator(circuit).data for the hand-made circuit and for
     QFTGate(5) and compare them with np.allclose. We got True.

Q17. What is Operator in qiskit?
A17. A class that gives the full unitary matrix (here 32 x 32 complex numbers) of a circuit
     without measurement. It is used to check circuits exactly.

Q18. Why do we need np.allclose rather than ==?
A18. Floating point numbers have tiny rounding errors (like 1e-16), so two correct matrices are
     never exactly equal; allclose allows a small tolerance.

Q19. What is the approximate QFT?
A19. A QFT that skips the very small rotation gates. It uses O(n log n) gates, and gives almost
     the same result. Good for noisy real hardware.

Q20. How does QFT help Shor's algorithm?
A20. Shor needs the period r of f(x) = a^x mod N. A register is put in a superposition that
     repeats with period r; the inverse QFT turns this repeating pattern into sharp peaks at
     multiples of 2^n / r, from which r is found.

Q21. How does phase estimation use the QFT?
A21. The phase of a unitary's eigenvalue is stored in the phases of a counting register; the
     inverse QFT converts it into a binary number that can be measured.

Q22. Is the QFT unitary? Why does it matter?
A22. Yes. Unitary means reversible and norm preserving, which every quantum gate must be. That
     is why an inverse QFT exists.

Q23. What does the barrier or decompose do here?
A23. decompose() only opens the QFTGate box so we can see the H, P and swap gates for the
     drawing. It does not change the result.

Q24. Can we read all Fourier coefficients from the QFT?
A24. No. We get only one random sample per shot, with probability |coefficient|^2. The QFT is
     only useful when the problem is designed so that the answer shows up as a high-probability
     outcome.

Q25. What is the difference between QFT and Hadamard transform?
A25. Applying H on every qubit is the Fourier transform over bit strings (groups of size 2).
     The QFT is the Fourier transform over the integers modulo N (it needs the extra phase
     rotations). For one qubit they are the same.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Change the code to encode the value 5 (binary 00101) instead. What do you edit?
A1. Change encode to qc.x([0, 2]). The input becomes 00101. After QFT then inverse QFT the
    output is 00101 in all shots.

Q2. What happens if you do not add qc.measure?
A2. No classical bits are written, so get_counts() has nothing to give and raises an error.

Q3. Write the line that applies the inverse QFT to all 5 qubits.
A3. qc.append(QFTGate(5).inverse(), range(5))

Q4. What happens if you remove the swap loop in the hand-made QFT?
A4. np.allclose returns False, because the output qubit order is reversed, so the matrix is
    different from the library QFT.

Q5. What would np.allclose return if you used the angle pi/2**(j-k+1) instead?
A5. False. The rotation angles would be wrong, so the unitary would be different.

Q6. How do you use 4 qubits instead of 5?
A6. Set n = 4 and change the X gates to fit (for example qc.x([0, 2])). The code uses n
    everywhere, and the hand-made loop adapts automatically.

Q7. How do you count how many gates the hand-made circuit has?
A7. hand.count_ops() gives for example {'cp': 10, 'h': 5, 'swap': 2}, and hand.size() gives
    the total (17).

Q8. How do you see the matrix of the QFT?
A8. np.round(Operator(QFTGate(n)).data, 2) prints the 32 x 32 matrix. Every entry has
    magnitude 1/sqrt(32) = 0.177.

Q9. What is the effect of changing shots to 100 in step 3?
A9. Fewer samples, so not all 32 outcomes may appear, and the bars look more uneven. The
    distribution stays flat on average.

Q10. How could you check that the QFT output really is flat without sampling?
A10. Use Statevector: Statevector(qc_without_measure).probabilities() gives all 32
     probabilities; each is 1/32 = 0.03125.

Q11. How can you apply the QFT to just the first 3 qubits of a 5-qubit circuit?
A11. qc.append(QFTGate(3), [0, 1, 2]); the second argument says which qubits to use.

END OF VIVA NOTES
'''
