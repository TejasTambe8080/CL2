"""
Practical 4 : Tackle Noise with Error Correction (3-qubit bit-flip code + readout mitigation)
=============================================================================================
INSTRUCTIONS
 1. Install:  pip install qiskit qiskit-aer numpy matplotlib
 2. Dataset : none. Everything runs on the local AerSimulator (no IBM account or API token).
              The manual's old IBMQ / Aer.get_backend('qasm_simulator') line maps to AerSimulator().
 3. Run     : python QAI_2_Error_Correction_Noise.py
 4. Steps   : noise model -> unprotected qubit -> 3-qubit repetition code -> sweep p
              -> plot vs theory 3p^2-2p^3 -> readout error demo (original vs mitigated counts)
"""

# --- Step 1: Import the libraries ---
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, ReadoutError
from qiskit.visualization import plot_histogram

SHOTS = 4000
SEED = 42

# --- Step 2: Build a noise model that flips a qubit (X error) with probability p ---
# The error is attached to the identity gate 'id', so we choose where the noise acts.
def bitflip_noise(p):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["id"])
    return nm

# --- Step 3: Circuit WITHOUT protection (one qubit, logical state |1>) ---
# Prepare |1>, let the noisy channel act, measure. Logical error = we read 0.
def unprotected_circuit():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.id(0)          # noisy channel
    qc.measure(0, 0)
    return qc

# --- Step 4: Circuit WITH the 3-qubit bit-flip repetition code ---
# q0,q1,q2 = data qubits, q3,q4 = ancilla (syndrome) qubits.
def protected_circuit():
    qc = QuantumCircuit(5, 1)
    # encode |1> -> |111>, noisy channel (id), syndromes s1 = q0^q1 and s2 = q1^q2,
    # corrections, read q0
    qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.id(0)
    qc.id(1)
    qc.id(2)
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    # s1=1,s2=1 -> flip q1
    qc.ccx(3, 4, 1)
    # s1=1,s2=0 -> flip q0
    qc.x(4)
    qc.ccx(3, 4, 0)
    qc.x(4)
    # s1=0,s2=1 -> flip q2
    qc.x(3)
    qc.ccx(3, 4, 2)
    qc.x(3)
    qc.measure(0, 0)
    return qc

# --- Step 5: Helper that runs a circuit with noise p and gives the error rate ---
def logical_error_rate(qc, p):
    sim = AerSimulator(noise_model=bitflip_noise(p))
    compiled = transpile(qc, sim, optimization_level=0)
    result = sim.run(compiled, shots=SHOTS, seed_simulator=SEED).result()
    counts = result.get_counts()
    # we encoded 1, so every '0' is an error
    return counts.get("0", 0) / SHOTS

print(protected_circuit().draw("text"))

# --- Step 6: Sweep p and compare unprotected, protected and theory ---
ps = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
unprot = [logical_error_rate(unprotected_circuit(), p) for p in ps]
prot = [logical_error_rate(protected_circuit(), p) for p in ps]
# >=2 of 3 qubits flipped
theory = [3 * p**2 - 2 * p**3 for p in ps]
print("\n    p   unprotected  protected  theory 3p^2-2p^3")
for p, u, c, t in zip(ps, unprot, prot, theory):
    print(f"{p:5.2f}   {u:9.4f}  {c:9.4f}  {t:9.4f}")

# --- Step 7: Plot the three curves ---
plt.plot(ps, unprot, "ro-", label="Unprotected")
plt.plot(ps, prot, "bs-", label="3-qubit code (simulated)")
plt.plot(ps, theory, "g--", label="Theory 3p^2 - 2p^3")
plt.xlabel("Physical error probability p")
plt.ylabel("Logical error rate")
plt.title("Bit-flip repetition code")
plt.legend()
plt.grid(True)
plt.show()

# --- Step 8: Measurement (readout) error demo, as in the manual ---
# Noise model: reading 0 gives 1 with prob 0.05, reading 1 gives 0 with prob 0.10.
ro = NoiseModel()
ro.add_all_qubit_readout_error(ReadoutError([[0.95, 0.05], [0.10, 0.90]]))
sim = AerSimulator(noise_model=ro)

# The manual's circuit: 3-qubit GHZ state (ideal result: 000 and 111, 50% each).
ghz = QuantumCircuit(3, 3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(0, 2)
ghz.measure([0, 1, 2], [0, 1, 2])
result = sim.run(transpile(ghz, sim), shots=SHOTS, seed_simulator=SEED).result()
counts = result.get_counts()
print("\nOriginal counts:", dict(sorted(counts.items())))

# --- Step 9: Calibration matrix: prepare each of the 8 basis states, measure ---
# M[measured, prepared]
M = np.zeros((8, 8))
for prepared in range(8):
    cal = QuantumCircuit(3, 3)
    for k in range(3):
        if (prepared >> k) & 1:
            cal.x(k)
    cal.measure([0, 1, 2], [0, 1, 2])
    result = sim.run(transpile(cal, sim), shots=SHOTS, seed_simulator=SEED).result()
    c = result.get_counts()
    for key, n in c.items():
        M[int(key, 2), prepared] = n / SHOTS

# --- Step 10: Mitigate: solve M * true = measured, clip negatives, renormalise ---
measured = np.zeros(8)
for i in range(8):
    measured[i] = counts.get(format(i, "03b"), 0) / SHOTS
true = np.clip(np.linalg.solve(M, measured), 0, None)
true = true / true.sum()
mitigated = {}
for i in range(8):
    mitigated[format(i, "03b")] = float(true[i])
shown = {}
for key, value in mitigated.items():
    if value > 0.001:
        shown[key] = round(value, 3)
print("Mitigated probabilities:", shown)
print("P(000)+P(111) original :", round(measured[0] + measured[7], 3))
print("P(000)+P(111) mitigated:", round(true[0] + true[7], 3))

# --- Step 11: Histogram of original vs mitigated ---
orig_prob = {}
for key, value in counts.items():
    orig_prob[key] = value / SHOTS
plot_histogram([orig_prob, mitigated], legend=["Original", "Mitigated"])
plt.show()






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 4 : TACKLE NOISE WITH ERROR CORRECTION
================================================================================

CODE EXPLANATION
----------------
1. Imports: numpy (calibration matrix maths), matplotlib (plots), QuantumCircuit and transpile
   from qiskit, AerSimulator from qiskit_aer (the local simulator), NoiseModel / pauli_error /
   ReadoutError to describe noise, plot_histogram to compare counts.
2. Noise model (bitflip_noise): a NoiseModel that applies a Pauli error "X with probability p,
   identity with probability 1-p" after every identity gate 'id'. We put an 'id' gate exactly
   where we want the noisy channel, so only that place is noisy and everything else is perfect.
3. Unprotected circuit: 1 qubit, X gate (state |1>), id (noisy), measure. If we read 0 the bit
   was flipped, so that is a logical error.
4. Protected circuit (3-qubit repetition code), 5 qubits:
   - q0,q1,q2 = data qubits, q3,q4 = ancilla qubits that hold the syndrome.
   - Encode: X on q0 gives |1>, then CX(0,1) and CX(0,2) copy it: |1> -> |111>.
   - Noisy channel: one 'id' on each data qubit (each can flip with probability p).
   - Syndrome: s1 = q0 xor q1 is stored in q3, s2 = q1 xor q2 is stored in q4.
   - Correction (Toffoli = ccx): s1=1,s2=1 means q1 flipped; s1=1,s2=0 means q0 flipped;
     s1=0,s2=1 means q2 flipped. The X gates before/after a ccx just turn a "control on 0"
     into a "control on 1" and are undone straight afterwards.
   - Decode: after correction all three data qubits agree, so measuring q0 gives the logical bit.
5. logical_error_rate: runs a circuit on AerSimulator(noise_model=...) with 4000 shots and
   seed_simulator=42, and returns (number of '0' results) / shots, because we encoded 1.
   optimization_level=0 stops transpile from deleting the 'id' gates.
6. Sweep: 8 values of p from 0.01 to 0.5. For each p we get unprotected, protected and the
   theory value 3p^2 - 2p^3, then print a table and plot all three curves.
7. Readout demo: a different noise model, ReadoutError([[0.95,0.05],[0.10,0.90]]), meaning
   0 is read as 1 with 5% and 1 is read as 0 with 10%. The circuit is the manual's 3-qubit
   GHZ circuit (H, CX, CX, measure). Ideal output is 000 and 111, 50% each.
8. Calibration matrix: for each of the 8 basis states 000..111 we prepare it with X gates,
   measure it, and fill column "prepared" of the 8x8 matrix M with the measured frequencies.
9. Mitigation: solve M * true = measured with numpy.linalg.solve, clip small negative values
   to 0 and renormalise. This replaces the old qiskit.ignis CompleteMeasFitter, which no
   longer exists. plot_histogram then draws Original vs Mitigated like the manual does.
10. Note: the manual's own code never defined a noise model and used removed functions
    (assemble, ignis, Aer from qiskit). This script keeps its intent (noise, then fix it)
    with modern qiskit 2.x and qiskit-aer, and adds real error correction (the repetition code).

EXPECTED OUTPUT (real run, shots = 4000, seed_simulator = 42)
----------------------------------------------------------------------
       p   unprotected  protected  theory 3p^2-2p^3
     0.01      0.0107     0.0008     0.0003
     0.05      0.0445     0.0057     0.0073
     0.10      0.0932     0.0238     0.0280
     0.20      0.1938     0.0980     0.1040
     0.30      0.2928     0.2175     0.2160
     0.40      0.3882     0.3528     0.3520
     0.45      0.4417     0.4270     0.4253
     0.50      0.4955     0.5040     0.5000

   Original counts: {'000': 1693, '001': 111, '010': 93, '011': 163, '100': 111,
                     '101': 167, '110': 162, '111': 1500}
   Mitigated probabilities: {'000': 0.488, '001': 0.001, '011': 0.003, '100': 0.003, '111': 0.505}
   P(000)+P(111) original : 0.798
   P(000)+P(111) mitigated: 0.993
   (The circuit text drawing is printed first. Two plots open: logical error vs p, and the
   Original vs Mitigated histogram. Protected beats unprotected for every p below 0.5; at
   p = 0.5 both are about 50% (pure guessing). Small differences from theory, such as 0.0008
   vs 0.0003, are sampling noise: 4000 shots give an error of about 0.5 to 1 percentage point
   on a rate near 0.5 and much less on small rates.)

OUTPUT GRAPHS EXPLAINED (and circuit diagrams for quantum scripts)
------------------------------------------------------------------
Each item: what it shows, how to read it, our result, a viva answer, a common mistake.
A. CIRCUIT DIAGRAM (text drawing, the 5-qubit repetition code)
   - Plots: q_0 to q_2 are data qubits, q_3, q_4 are ancilla (check) qubits; "c: 1/" = 1 bit.
   - Symbols: [ X ] = NOT; dot joined to [ X ] = CNOT; two dots joined to [ X ] = Toffoli.
   - [ I ] does nothing; it only marks where noise (X error, probability p) is attached.
   - [M] = measure q_0 into the classical bit.
   - Read it: X on q_0, two CNOTs copy it to |111>, then three [ I ] noise boxes hit the data.
   - Then 4 CNOTs write the parity checks (syndrome) to q_3, q_4; 3 Toffolis fix the flip.
   - Ours: this circuit gives logical error 0.0238 at p = 0.10 (see Figure 1).
   - If asked: "The ancillas show WHICH qubit flipped; Toffoli flips it back."
   - Misreading: [ I ] boxes are not errors by themselves, only noise markers.

B. FIGURE 1 - LINE PLOT "Bit-flip repetition code"
   - Plots: logical error rate (how often we read the wrong bit) against physical error p.
   - x axis: physical error probability p (0.01 to 0.5); y axis: logical error rate (0 to 0.5).
   - Legend: red circles = unprotected; blue squares = 3-qubit code run; green dashed = theory.
   - Read it: a lower curve means fewer errors. The red line is straight (error = p).
   - Blue bends below red, so the code helps; where the curves meet, it stops helping.
   - Ours: at p = 0.10, unprotected 0.0932 vs protected 0.0238; both near 0.5 at p = 0.50.
   - If asked: "It fails only if 2 or more qubits flip: 3p^2 - 2p^3, below p for p < 0.5."
   - Misreading: the code is not perfect; the small blue-green gap is shot noise.

C. FIGURE 2 - HISTOGRAM "Original vs Mitigated" (readout error demo)
   - Plots: 3-qubit GHZ results with noisy readout, before (Original) and after (Mitigated).
   - x axis: 3-bit outcomes 000 to 111; y axis: quasi-probability (fraction of shots).
   - Legend: blue = Original, orange = Mitigated. Numbers on the bars are the exact values.
   - Read it: perfect GHZ = two tall bars, 000 and 111, near 0.5 each; other bars are errors.
   - Ours: P(000)+P(111) rises from 0.798 to 0.993; the stray bars shrink to 0.001 to 0.003.
   - If asked: "We measure how the device misreads each state in a matrix, then invert it."
   - Misreading: it fixes only readout errors, not gate errors; missing bars were clipped to 0.

DETAILED THEORY NOTES
---------------------
1. WHAT IS QUANTUM ERROR CORRECTION (AND NOISE)?
   - Noise: real qubits are fragile. Interaction with the environment (decoherence), imperfect
     gates and imperfect measurement change the state in an unwanted way.
   - Error correction: store one logical qubit in several physical qubits so that an error
     on one of them can be detected and undone without destroying the information.
   - Intuition: send each bit three times and take a majority vote. If one copy flips, the
     other two outvote it.
   - Quantum twist: we cannot copy an unknown state (no-cloning theorem) and measuring the data
     destroys superposition. So the code spreads the information by entanglement and only
     measures the SYNDROME (parities of pairs), which tells where an error is but nothing
     about the stored value.
   - Maths in words: with flip probability p on each of 3 qubits, the majority vote fails only
     if 2 or 3 qubits flip: 3 p^2 (1-p) + p^3 = 3p^2 - 2p^3. For small p this is about 3p^2,
     much smaller than p. Break-even is p = 0.5.

2. TYPES AND VARIANTS
   - Bit-flip code (this practical): protects against X errors only.
   - Phase-flip code: same idea in the Hadamard basis, protects against Z errors.
   - Shor 9-qubit code: combines both, corrects any single-qubit error.
   - Steane 7-qubit code (CSS code) and the 5-qubit perfect code.
   - Surface code / topological codes: many qubits on a 2-D grid; leading candidate for
     large scale machines.
   - Error MITIGATION (not correction): methods such as measurement (readout) mitigation,
     zero-noise extrapolation and probabilistic error cancellation. They fix results by
     post-processing and need no extra encoded qubits. The last part of this practical is
     this type.

3. KEY CONCEPTS AND TERMS
   - Physical qubit : a real qubit (noisy). Logical qubit : the protected qubit made from
     several physical ones.
   - Bit flip : X error, 0 <-> 1. Phase flip : Z error, changes the sign of |1>.
   - Repetition code : |0> -> |000>, |1> -> |111> (by two CNOT gates).
   - Syndrome : the parity checks (q0 xor q1, q1 xor q2). 00 = no error, 10 = q0, 11 = q1,
     01 = q2 flipped.
   - Ancilla qubit : helper qubit used to extract the syndrome without measuring the data.
   - Toffoli (ccx) : controlled-controlled-NOT; flips the target only if both controls are 1.
   - Logical error rate : probability that the decoded logical bit is wrong.
   - Noise model : a description of errors attached to gates and measurements in the simulator.
   - Readout (measurement) error : the qubit is correct but is read wrongly.
   - Calibration matrix M : M[i][j] = probability of measuring i when j was prepared.
   - Threshold : the physical error rate below which encoding helps. Here it is p = 0.5 for
     this toy model (real codes have thresholds near 1% because their gates are noisy too).
   - Shots : number of repeated runs; seed_simulator : fixes the simulator random numbers.

4. HOW TO JUDGE THE RESULT
   - Compare the logical error rate with the unprotected rate for the same p. The protected
     value must be lower for p < 0.5. Table above: at p = 0.1, 0.0932 -> 0.0238.
   - Compare with theory 3p^2 - 2p^3; differences should be within sampling noise
     (about sqrt(p(1-p)/4000), e.g. 0.005 at p = 0.1).
   - For mitigation: P(000)+P(111) should move closer to the ideal 1.0. Here 0.798 -> 0.993.
   - Not valid here: accuracy / confusion matrix (no classifier). The measure is the logical
     error rate.

5. SETUP (NO DATA SPLIT)
   - No train/test split: this is a simulation, not machine learning.
   - Setup values: 4000 shots per point, seed 42, p swept from 0.01 to 0.5, logical state |1>.
   - Only the 'id' gate is noisy, so the ancilla and CNOT/Toffoli gates are perfect. This is
     the standard textbook assumption and is why the code works so well.

6. THE SIMULATOR
   - AerSimulator (qiskit_aer) is a fast classical simulator of quantum circuits. With a
     NoiseModel it imitates a noisy device. It runs locally: no IBM account, no API token.
   - The manual's IBMQ / Aer.get_backend('qasm_simulator') becomes AerSimulator().
     A real device would be reached with qiskit_ibm_runtime (needs a token); not needed here.
   - pauli_error([("X", p), ("I", 1-p)]) builds the bit-flip channel. ReadoutError takes
     rows [P(read 0|true 0), P(read 1|true 0)] and [P(read 0|true 1), P(read 1|true 1)].
   - ignis (old qiskit) is removed. Its job (measurement calibration) is done here by hand with
     numpy, which is also easier to explain.

7. ADVANTAGES
   - The repetition code is simple: 3 qubits, 2 CNOT to encode, a few gates to correct.
   - Cuts the error from p to about 3p^2 when p is small.
   - Readout mitigation needs no extra qubits, only a few calibration runs.
   - Good teaching example of syndrome measurement without reading the data.

8. DISADVANTAGES
   - Bit-flip code cannot fix phase errors; real noise has both (need Shor/Steane/surface codes).
   - Costs 3x qubits plus ancillas, and extra gates that are themselves noisy in real devices.
   - Above p = 0.5 (and near it) the code gives no benefit.
   - Calibration matrix grows as 2^n x 2^n, so full inversion is impractical for many qubits
     (tensored or matrix-free methods are used). Inversion can give negative probabilities
     (hence the clip and renormalise step).
   - Mitigation lowers bias but increases variance and does not fix gate errors.

9. REAL-WORLD USES
   - Building fault-tolerant quantum computers (superconducting, ion trap and other hardware).
   - Quantum memory and quantum communication (repeaters).
   - Readout mitigation is used routinely on cloud hardware before reading results.
   - Classical analogues: ECC memory, CD/DVD Reed-Solomon codes, deep-space links.

10. COMPARISON WITH RELATED METHODS
   - Error correction vs error mitigation : correction encodes information and fixes errors
     during the run (needs many qubits); mitigation post-processes results (no overhead
     qubits, but cannot scale to deep circuits).
   - Repetition code vs Shor code : repetition fixes one error type, Shor fixes any single-qubit
     error with 9 qubits.
   - Quantum vs classical repetition : classically you can copy bits and read them; quantum
     cannot copy or look directly, so parity checks (syndrome) are used.
   - Surface code vs repetition : surface code handles X and Z and has a real threshold near
     1% but needs hundreds to thousands of physical qubits per logical qubit.

LIBRARIES AND FUNCTIONS USED
----------------------------
- numpy (np) : np.zeros makes the 8x8 matrix; np.array builds the measured vector;
  np.linalg.solve solves M x = measured; np.clip(x, 0, None) removes negatives; .sum() normalises.
- matplotlib.pyplot (plt) : plot (lines), xlabel/ylabel/title/legend/grid, show.
- qiskit.QuantumCircuit(n_qubits, n_bits) : the circuit; methods x, h, cx, ccx, id, measure, draw("text").
- qiskit.transpile(qc, sim, optimization_level=0) : adapts the circuit to the simulator; level 0
  keeps our 'id' gates.
- qiskit_aer.AerSimulator(noise_model=...) : local simulator; .run(circuit, shots=, seed_simulator=)
  then .result().get_counts() gives a dictionary like {'000': 1693, ...}.
- qiskit_aer.noise.NoiseModel : container for errors; add_all_qubit_quantum_error(error, ["id"])
  and add_all_qubit_readout_error(error).
- qiskit_aer.noise.pauli_error : channel from a list of (Pauli, probability).
- qiskit_aer.noise.ReadoutError : 2x2 matrix of read probabilities.
- qiskit.visualization.plot_histogram(list, legend=[...]) : bar chart of several count/probability sets.
- int(key, 2) : turns a bit string such as '101' into the number 5 (matrix index).
- format(i, "03b") : turns 5 into '101'.

DETAILED VIVA Q&A
-----------------
Q1. How can we adapt error correction codes to different types of noise? How can we design
    error correction codes that are scalable to large numbers of qubits?
A1. First find the dominant noise: bit flips (X), phase flips (Z) or both. Use the bit-flip
    code for X noise, the phase-flip code for Z noise, and a combined code (Shor, Steane,
    surface) when both exist. Biased-noise codes (e.g. XZZX surface code) are tuned when one
    type is much stronger. For scaling, use codes whose size grows with a code distance and
    whose checks are local, like the surface code, which has a threshold: below it, adding
    more qubits makes the logical error smaller.

Q2. How can we implement error correction in a way that minimizes overhead?
A2. Use codes with a high rate (more logical qubits per physical qubit), such as LDPC codes;
    use fewer syndrome rounds; choose the smallest distance that meets the target error rate;
    use better decoders; reduce physical error rates so smaller codes are enough; and use
    error mitigation for shallow circuits where full correction is not needed.

Q3. How can we combine error correction with other techniques, such as quantum feedback control,
    to improve the performance of quantum devices?
A3. The syndrome is measured and the result is fed back to apply a correction (that is
    feedback control; here done by the ccx gates, on hardware by mid-circuit measurement and
    classically controlled gates). It can be combined with dynamical decoupling, pulse-level
    calibration, error mitigation and real-time decoders to lower the effective error rate.

Q4. What is quantum noise?
A4. Any unwanted change of a qubit state caused by the environment (decoherence), imperfect
    gates, crosstalk or faulty measurement.

Q5. What is a bit-flip error?
A5. An X error that swaps |0> and |1>. In the code it is the pauli_error with X probability p.

Q6. Explain the 3-qubit repetition code.
A6. Encode a|0> + b|1> as a|000> + b|111> using two CNOTs. If at most one qubit flips,
    parity checks find which one and a correction undoes it. It fails only if two or more
    qubits flip.

Q7. What is a syndrome?
A7. The result of the parity checks (q0 xor q1, q1 xor q2). It points to the flipped qubit
    but reveals nothing about the logical value, so the superposition survives.

Q8. Why can we not just copy the qubit three times?
A8. The no-cloning theorem forbids copying an unknown quantum state. The CNOTs do not copy
    the state; they entangle the qubits to form a|000> + b|111>.

Q9. Why are ancilla qubits used?
A9. To read the parities without measuring the data qubits. Measuring the data would collapse
    the superposition.

Q10. What does the Toffoli (ccx) gate do in the correction step?
A10. It flips the target data qubit only when both syndrome bits are 1. With X gates around it
     we can select the other syndrome patterns (10, 01).

Q11. Why is the protected logical error rate 3p^2 - 2p^3?
A11. Majority vote fails when 2 qubits flip (3 ways, p^2 (1-p)) or 3 qubits flip (p^3):
     3p^2(1-p) + p^3 = 3p^2 - 2p^3.

Q12. When is the code useful?
A12. When 3p^2 - 2p^3 < p, i.e. p < 0.5. At p = 0.5 they are equal and above it the code makes
     things worse. Our table shows 0.0932 vs 0.0238 at p = 0.1 but 0.4955 vs 0.5040 at 0.5.

Q13. Why does the simulated value differ a little from theory?
A13. Only 4000 shots are used, so the estimate has sampling noise (for example 0.0057
     measured vs 0.0073 theory at p = 0.05).

Q14. What is the difference between error correction and error mitigation?
A14. Correction uses extra qubits to detect and fix errors during the computation. Mitigation
     keeps the circuit as is and corrects the final statistics by post-processing.

Q15. What is measurement (readout) error?
A15. The qubit is in the right state but the detector reports the wrong bit, for example
     reading 1 as 0 with 10% probability in our model.

Q16. How does the calibration matrix work?
A16. We prepare every basis state and record what is measured. Column j of M holds the read
     probabilities for state j. Then measured = M x true, so true = M^-1 x measured.

Q17. Why do we clip and renormalise after solving?
A17. Because of shot noise the inverse can give small negative numbers, which are not valid
     probabilities. We set them to 0 and divide by the sum so the total is 1.

Q18. What result did mitigation give?
A18. The GHZ outcomes 000 and 111 had total probability 0.798 before and 0.993 after; the
     ideal value is 1.0.

Q19. Why is the GHZ circuit used?
A19. It is the manual's circuit. Ideally only 000 and 111 appear (50% each), so any other
     outcome is clearly caused by noise.

Q20. Why does the manual's code not show any noise?
A20. It never defines a noise model, so the simulator is ideal and mitigation changes almost
     nothing. We add an explicit ReadoutError so there is real noise to remove.

Q21. Why is optimization_level=0 used in transpile?
A21. At higher levels the transpiler may remove identity gates as useless, and then the noise
     attached to 'id' would disappear.

Q22. What is a logical qubit and a physical qubit?
A22. Physical qubits are the real noisy ones. A logical qubit is the protected information
     stored across several physical qubits (3 data qubits here).

Q23. What is the threshold theorem?
A23. If the physical error rate is below a threshold, arbitrarily long computation can be made
     reliable by adding more encoding layers, at a moderate (polylogarithmic) cost.

Q24. Can the bit-flip code correct a phase error?
A24. No. A Z error does not change the Z-basis parities. A phase-flip code (same circuit in the
     Hadamard basis) or the Shor code is needed.

Q25. Why use seed_simulator?
A25. It fixes the random number stream of the simulator so the counts are repeatable.

Q26. What happens if two qubits flip?
A26. The syndrome points to the wrong qubit, the "correction" flips a good one, and all three
     qubits end up wrong. This is the logical error that occurs with probability about 3p^2.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Change p to 0.001. What happens to the logical error rate?
A1. Unprotected is about 0.001, protected about 3 x 10^-6, so nearly no errors appear in 4000
    shots (probably 0 counts). The gain is larger when p is smaller.

Q2. Change the logical state to |0>. What must change in the code?
A2. Remove qc.x(0) and count '1' instead of '0' as the error in logical_error_rate
    (counts.get("1", 0)).

Q3. Write the line that adds bit-flip noise with probability p to the id gate.
A3. nm.add_all_qubit_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["id"])

Q4. What happens if you set optimization_level to 3 in the sweep?
A4. The transpiler may drop the id gates, so no noise is applied and both rates become 0.

Q5. Increase the shots from 4000 to 40000. What changes?
A5. The curves get smoother and match theory more closely; run time is longer. Sampling
    error falls by a factor of about 3.

Q6. Add a second round of noise (another id on each data qubit before the syndrome). Effect?
A6. Each qubit now flips with probability 2p(1-p), so the effective p is larger and the
    protected rate is about 3 (2p)^2 for small p.

Q7. Make the ancilla qubits noisy too. What do you expect?
A7. Wrong syndromes cause wrong corrections, so the protected error rate rises and the
    threshold moves well below 0.5 (the code would only help for small p).

Q8. Write the line that builds the readout error for 2% (0->1) and 5% (1->0).
A8. ReadoutError([[0.98, 0.02], [0.05, 0.95]])

Q9. How would you change the readout demo to use 2 qubits only?
A9. Use QuantumCircuit(2, 2), measure [0, 1], build a 4x4 matrix (range(4), format "02b",
    range(2) for the X loop) and use 4 instead of 8 in the np.zeros, loops and solve.

Q10. Which lines make the syndrome bit s1 and which make s2?
A10. s1: qc.cx(0, 3), qc.cx(1, 3). s2: qc.cx(1, 4), qc.cx(2, 4).

Q11. How do you plot only the protected curve and the theory curve?
A11. Delete the plt.plot line for unprot (or comment it) and keep the other two plt.plot calls.

Q12. What if np.linalg.solve raises an error?
A12. The calibration matrix is singular (for example a read error of 50%). With realistic errors
     (5% and 10%) M is invertible. np.linalg.lstsq is a safer alternative.

END OF VIVA NOTES
'''
