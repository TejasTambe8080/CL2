"""
Practical 2 : Quantum Teleportation (3 qubits)
==============================================
INSTRUCTIONS
 1. Install:  pip install qiskit qiskit-aer matplotlib
 2. Dataset : None needed. Everything runs on qiskit's local AerSimulator
              (no IBM account/token).
 3. Run     : python QAI_4_Quantum_Teleportation.py
 4. Steps   : prepare message on q0 -> Bell pair q1,q2 -> Bell measurement on q0,q1
              -> correct q2 -> measure -> test 1: teleport |1>
              -> test 2: teleport an arbitrary state
"""

# --- Step 1: Import the libraries ---
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

sim = AerSimulator(seed_simulator=42)

# --- Step 2: A small helper that builds the teleportation circuit ---
# q0 = message (Alice), q1 = Alice's half of the Bell pair, q2 = Bob's half.
def teleport_circuit(prepare, undo=None):
    qc = QuantumCircuit(3, 3)
    # message on q0, Bell pair q1-q2, Alice's basis change, Bob's corrections
    # then an optional undo step
    prepare(qc)
    qc.barrier()
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier()
    qc.cx(1, 2)                     # X if q1=1
    qc.cz(0, 2)                     # Z if q0=1
    qc.barrier()
    if undo is not None:
        undo(qc)                    # test 2 only
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc

# --- Step 3: Test 1 - teleport the state |1> (X gate on q0) ---
qc1 = teleport_circuit(prepare=lambda qc: qc.x(0))
print(qc1.draw(output="text"))

# --- Step 4: Run 1024 shots and look at the counts ---
job1 = sim.run(transpile(qc1, sim), shots=1024)
counts1 = job1.result().get_counts()
print("Test 1 counts (bits are q2 q1 q0):", counts1)
# Bob's qubit q2 is the leftmost bit; count how often it is 1.
bob1 = sum(c for bits, c in counts1.items() if bits[0] == "1")
print("Bob measured 1 in", bob1, "of 1024 shots")

# --- Step 5: Test 2 - teleport an arbitrary state made by ry(theta) and rz(phi) ---
theta = 1.1                     # demo input angles
phi = 0.7
def prepare(qc):
    qc.ry(theta, 0)
    qc.rz(phi, 0)

def undo(qc):                       # inverse of prepare, in reverse order, on q2
    qc.rz(-phi, 2)
    qc.ry(-theta, 2)

qc2 = teleport_circuit(prepare, undo)
job2 = sim.run(transpile(qc2, sim), shots=1024)
counts2 = job2.result().get_counts()
print("Test 2 counts (bits are q2 q1 q0):", counts2)
bob2 = sum(c for bits, c in counts2.items() if bits[0] == "0")
print("After undoing the preparation Bob measured 0 in", bob2, "of 1024 shots", end=" ")
print("(0 means the state arrived)")

# --- Step 6: Plot both histograms ---
# Draw on our own axes (ax=) so the chart also shows inside Jupyter.
fig, ax = plt.subplots(figsize=(8, 4))
labels = ["Test 1: teleport |1>", "Test 2: arbitrary state"]
plot_histogram([counts1, counts2], legend=labels, ax=ax)
plt.show()






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 2 : QUANTUM TELEPORTATION
================================================================================

CODE EXPLANATION
----------------
1. Imports: QuantumCircuit and transpile (qiskit), AerSimulator (qiskit_aer), plot_histogram
   (qiskit.visualization) and matplotlib to show the plot.
2. sim = AerSimulator(seed_simulator=42): local simulator with a fixed seed so the counts repeat.
   The manual used Aer.get_backend('qasm_simulator') with execute(); execute() is removed in
   modern qiskit, so sim.run(transpile(...)) is used. On real hardware you would use qiskit-ibm-runtime.
3. Helper teleport_circuit(prepare, undo): builds a circuit with 3 qubits and 3 classical bits.
   q0 = the message qubit (Alice), q1 = Alice's half of the Bell pair, q2 = Bob's half.
   Steps inside the helper:
     a) prepare(qc)            : puts the message state on q0 (test 1: X gate; test 2: ry, rz).
     b) h(1), cx(1, 2)         : makes the Bell pair (q1 and q2 become entangled).
     c) cx(0, 1), h(0)         : Alice's Bell measurement basis change on q0 and q1.
     d) cx(1, 2), cz(0, 2)     : Bob's corrections. X on q2 if q1 is 1 and Z on q2 if q0 is 1.
        This is the "deferred measurement" form: the classical control is replaced by
        quantum controlled gates. (The textbook way measures q0, q1 first and then sends the
        two classical bits to Bob. The old c_if method was removed, but qiskit 2.x has
        "with qc.if_test((clbit, 1)): qc.x(2)" for real classically-controlled corrections
        after measurement. This script uses the deferred form, which is equivalent and simpler.)
     e) undo(qc)               : used only in test 2 to reverse the preparation on q2.
     f) measure all three qubits into the three classical bits.
   barrier() only draws a dashed line to separate the stages; it does not change the logic.
4. Test 1 teleports |1> (qc.x(0)). The expected result is that Bob's qubit q2 is always 1.
   Counts are read as 'q2 q1 q0' (left bit is q2). q0 and q1 are random (4 outcomes of about
   25 percent each) but q2 is always 1. This is the proof of teleportation.
5. Test 2 teleports an arbitrary state: ry(1.1) then rz(0.7) on q0. We cannot see an arbitrary
   state in a measurement, so we apply the INVERSE preparation (rz(-0.7), ry(-1.1)) on q2. If
   q2 really holds the same state, the inverse brings it back to |0>, so q2 reads 0 always.
6. plot_histogram draws both count sets side by side.

EXPECTED OUTPUT (real run, AerSimulator, seed_simulator=42)
----------------------------------------------------------------------
   (circuit drawing: X on q0, H and CX for the Bell pair, CX and H on q0, then CX and CZ on q2,
    then three measurements)
   Test 1 counts (bits are q2 q1 q0): {'101': 276, '100': 234, '110': 256, '111': 258}
   Bob measured 1 in 1024 of 1024 shots
   Test 2 counts (bits are q2 q1 q0): {'010': 264, '011': 234, '000': 257, '001': 269}
   After undoing the preparation Bob measured 0 in 1024 of 1024 shots (0 means the state arrived)
   (Left bit is always 1 in test 1 and always 0 in test 2. The other two bits are random, about
   256 each, because Alice's measurement results are random. Exact counts can differ a little
   on another machine.)

OUTPUT GRAPHS EXPLAINED (and circuit diagrams)
----------------------------------------------
Two outputs: the circuit drawing and one histogram.

A. CIRCUIT DIAGRAM (text drawing of Test 1)
   - What it shows: the teleportation circuit as text; time runs from left to right.
   - Wires: q_0 = message, q_1 and q_2 = Bell pair; "c: 3/" = the 3 classical bits.
   - Symbols: [X] flip, [H] Hadamard, dot + box = CX, two dots = CZ, [M] measure.
   - Dashed barrier lines only separate the stages; they change nothing.
   - Read in blocks: message | Bell pair | Alice's CX and H | Bob's CX and CZ | measure.
   - Our result: X on q0, H and CX, CX and H on q0, CX and CZ on q2, 3 measurements.
   - If asked: "Block 1 is the message, 2 the entangled pair, 3 Alice measures, 4 Bob fixes."
   - Common misreading: a lone dot is half of a gate; q_0 is the top wire, LAST count bit.

B. FIGURE 1 - histogram of both tests (blue = Test 1, orange = Test 2)
   - What it plots: how often each 3-bit result came up in 1024 shots, for both tests.
   - X axis: bit string "q2 q1 q0" (left bit = Bob's q2). Y axis: Count of shots.
   - The number above each bar is its exact count; the legend is at the top right.
   - Tall bar = common result. Four equal bars (about 256 each) = even spread.
   - Check only the LEFT bit: all blue bars start with 1, all orange bars start with 0.
   - Our result: Bob measured 1 in 1024 of 1024 shots (Test 1), 0 in 1024 of 1024 (Test 2).
   - If asked: "The left bit never changes, so q0's state reached q2; the rest is random."
   - Common misreading: four bars per colour is correct; one single tall bar is not expected.

DETAILED THEORY NOTES
---------------------
1. WHAT IS QUANTUM TELEPORTATION?
   - Definition: a protocol that transfers an unknown quantum state from Alice to Bob using one
     shared entangled pair and two classical bits. The qubit itself does not travel; only its
     state is recreated at Bob's side and destroyed at Alice's side.
   - Type: quantum communication protocol (a quantum circuit). No learning involved.
   - Intuition: Alice and Bob share a "magic pair". Alice entangles her message qubit with her
     half, measures both, and phones two ordinary bits to Bob. Bob uses those two bits to fix
     his half so that it becomes exactly Alice's original state.
   - Maths in words: message |psi> = a|0> + b|1>. After the Bell pair and Alice's operations the
     total state is an equal mix of four terms, one for each result of Alice's measurement
     (00, 01, 10, 11). In each term Bob's qubit is |psi> after a simple change: nothing (00),
     X (01), Z (10) or ZX (11). So Bob applies X if the second bit is 1 and Z if the first bit
     is 1, and he then holds |psi>.
   - No cloning and no faster-than-light: the original is destroyed by measurement, so the
     no-cloning theorem is not broken. Bob cannot use his qubit until the two classical bits
     arrive, so no information travels faster than light.

2. TYPES AND VARIANTS
   - Standard teleportation (this practical) with classical bits sent from Alice to Bob.
   - Deferred measurement version (used here): controlled gates instead of classical control.
   - Entanglement swapping : teleporting one half of an entangled pair; basis of quantum
     repeaters.
   - Gate teleportation : teleporting the effect of a gate; used in fault-tolerant quantum
     computing.
   - Superdense coding : the "opposite" protocol; sends 2 classical bits using 1 qubit and one
     shared entangled pair.
   - Real experiments : done with photons, ions and satellites (Micius, over 1000 km).

3. KEY CONCEPTS AND TERMS
   - Entanglement : a state of two or more qubits that cannot be described one by one. Measuring
     one instantly fixes what you will see on the other. Example: Bell state
     (|00> + |11>)/sqrt(2).
   - Bell pair / EPR pair : the two-qubit maximally entangled state made by H on q1 then CX
     q1 -> q2.
   - Bell measurement : measuring two qubits in the Bell basis; done by CX, H and then a normal
     measurement.
   - Hadamard (H) : makes superposition. CNOT / CX : flips the target if the control is 1.
     X gate : bit flip. Z gate : phase flip (changes the sign of |1>). CZ : controlled Z.
   - ry(theta), rz(phi) : rotations about the Y and Z axes of the Bloch sphere; with these two
     we can make any single-qubit state.
   - Classical channel : ordinary communication (phone, internet) used to send the two bits.
   - Deferred measurement principle : measuring a qubit and then using the bit as a control is
     the same as using the qubit directly as a quantum control and measuring at the end.
   - Barrier : a visual separator in the circuit drawing.
   - Qiskit bit order : in the printed counts the leftmost bit is the highest numbered qubit
     (q2), the rightmost is q0.

4. HOW TO JUDGE THE RESULT
   - Test 1: q2 must be 1 in every shot (1024 of 1024). The other two bits should be about
     uniformly random (4 outcomes, about 256 each).
   - Test 2: after undoing the preparation, q2 must be 0 in every shot. Even a small leftover
     (for example 5 percent ones) would mean the state was not teleported exactly.
   - On a noise-free simulator the result is exact (fidelity 1). On a real device noise would
     give some wrong results (for example 1 to 10 percent).
   - Why the second test is needed: measuring |1> only tests one basis state. The inverse
     preparation test checks a full arbitrary state, including its phase.

5. DATA SPLIT / RELEVANT SETUP
   - No data and no train/test split. Settings are only: the message state (X gate or ry/rz
     angles), shots (1024) and the seed.
   - Demo input: theta = 1.1, phi = 0.7 (any angles work).

6. THE SIMULATOR (AerSimulator)
   - qiskit_aer.AerSimulator simulates the 3 qubits exactly on the classical computer (8
     amplitudes). It is noise free by default.
   - seed_simulator=42 repeats the random measurement outcomes.
   - No account or token is needed. The manual's Aer.get_backend('qasm_simulator') plus execute() becomes AerSimulator plus run().

7. ADVANTAGES
   - Moves an unknown quantum state without sending the qubit through a noisy channel.
   - Needs only classical communication plus pre-shared entanglement.
   - Does not violate no-cloning (the original is destroyed).
   - Basis of quantum networks, repeaters and distributed quantum computing.

8. DISADVANTAGES
   - Needs a pre-shared entangled pair for every qubit teleported.
   - Needs two classical bits per qubit, so it cannot be faster than light.
   - Entanglement is fragile; distance and noise reduce fidelity.
   - Bell measurement is hard to do with high success on some hardware (for example photons).
   - It transfers the state, it does not copy it (the sender loses it).

9. REAL-WORLD USES
   - Quantum internet and quantum repeaters.
   - Linking separate quantum processors (distributed quantum computing).
   - Gate teleportation inside fault-tolerant quantum computers.
   - Quantum cryptography networks.
   - Satellite experiments (China's Micius teleported photon states over 1400 km).

10. COMPARISON WITH RELATED METHODS
   - Teleportation vs classical copy : a classical file can be copied and the original kept;
     a qubit state cannot be copied (no cloning) and is destroyed at the source.
   - Teleportation vs sending the qubit directly : direct sending risks losing the qubit in a
     noisy channel; teleportation needs only classical bits after the entanglement is shared.
   - Teleportation vs superdense coding : teleportation sends 1 qubit state with 2 classical
     bits; superdense coding sends 2 classical bits with 1 qubit.
   - Teleportation vs quantum key distribution (BB84) : teleportation moves a quantum state;
     BB84 only creates a shared secret key.

LIBRARIES AND FUNCTIONS USED
----------------------------
- qiskit.QuantumCircuit(3, 3) : circuit with 3 qubits and 3 classical bits.
- qc.x(q), qc.h(q) : X (bit flip) and Hadamard gates. qc.cx(c, t) : CNOT with control c and
  target t. qc.cz(c, t) : controlled-Z. qc.ry(angle, q), qc.rz(angle, q) : rotations.
- qc.barrier() : a visual divider; no effect on results.
- qc.measure([0, 1, 2], [0, 1, 2]) : measure qubit i into classical bit i.
- qc.draw(output="text") : ASCII drawing of the circuit.
- qiskit.transpile(qc, sim) : converts the circuit to gates the simulator supports.
- qiskit_aer.AerSimulator(seed_simulator=42) : local simulator, fixed seed.
- sim.run(circuit, shots=1024).result().get_counts() : run and get the dictionary of counts.
- qiskit.visualization.plot_histogram(list_of_counts, legend=[...]) : bar chart of counts, one
  colour per dictionary.
- matplotlib.pyplot.show() : opens the figure window.
- def prepare(qc) / def undo(qc) : small functions passed to teleport_circuit; they add the
  message-preparation gates and the inverse gates to the circuit they receive.

DETAILED VIVA Q&A
-----------------
Q1. What is entanglement?
A1. Entanglement is a quantum link between two or more qubits so that their joint state cannot
    be written as separate states. Measuring one qubit immediately tells you the result of the
    other. Example: the Bell state (|00> + |11>)/sqrt(2), where both qubits always give the same
    result, but each result alone is random.

Q2. What are different gates used in teleportation?
A2. Hadamard (H) to create superposition, CNOT (CX) to entangle and to do the Bell
    measurement, and the Pauli X and Z gates (here CX and CZ controlled versions) for Bob's
    corrections. The X gate (or ry and rz) prepares the message state. Measurement is used at
    the end.

Q3. What is teleportation?
A3. Quantum teleportation is a protocol that sends an unknown quantum state from one place to
    another using a shared entangled pair and two classical bits. The state is recreated at the
    receiver and destroyed at the sender; no matter or qubit travels between them.

Q4. How Quantum circuit is designed?
A4. First decide the qubits and classical bits (3 and 3). Then add gates in order: prepare the
    message on q0; make a Bell pair on q1 and q2 (H then CX); entangle the message with Alice's
    half (CX q0 -> q1, then H on q0); apply Bob's corrections (CX q1 -> q2 and CZ q0 -> q2);
    finally measure. The circuit is then transpiled and run on a backend with many shots.

Q5. Why do we need two classical bits?
A5. Alice's Bell measurement has four possible results (00, 01, 10, 11), and each needs a
    different correction by Bob (nothing, X, Z, or both). Two bits are needed to tell Bob which
    one to use.

Q6. Does teleportation allow faster-than-light communication?
A6. No. Bob's qubit is useless until the two classical bits arrive, and those travel at normal
    speed. Without them Bob's qubit looks random.

Q7. Does teleportation violate the no-cloning theorem?
A7. No. Alice's measurement destroys the original state at her side, so only one copy ever
    exists.

Q8. How is the Bell pair created?
A8. H on q1 makes (|0> + |1>)/sqrt(2); then CX with q1 as control and q2 as target gives
    (|00> + |11>)/sqrt(2). One half goes to Alice and one half to Bob.

Q9. Why is q2 always 1 in test 1?
A9. The message was |1> (X on q0). After the corrections Bob's qubit holds exactly that state
    so measuring it always gives 1. The other two bits are random because they are Alice's
    measurement results.

Q10. Why do we use the inverse preparation in test 2?
A10. An arbitrary state a|0> + b|1> cannot be seen directly in one measurement. If we apply the
     exact inverse of the preparation circuit on Bob's qubit, a correct state returns to |0>.
     Then measuring 0 in all shots proves both the probabilities and the phase arrived.

Q11. What is the role of the CZ gate?
A11. It is the phase correction. If Alice's q0 result is 1, Bob's qubit has a sign flip on |1>,
     and Z fixes it. CZ applies Z on q2 when q0 is 1.

Q12. What is the role of the CX from q1 to q2 in the last step?
A12. It is the bit flip correction. If Alice's q1 result is 1, Bob's qubit is flipped, and X
     fixes it. CX applies X on q2 when q1 is 1.

Q13. What is the deferred measurement principle?
A13. A measurement followed by classical control of a gate can be replaced by a quantum
     controlled gate followed by the measurement at the end. The final statistics are the same.
     We use it because it is simpler than classical control (see the coding Q12 below).

Q14. What is a Bell measurement?
A14. A joint measurement of two qubits in the basis of the four Bell states. In a circuit it is
     done by CX (control first qubit, target second), H on the first qubit and then normal
     measurement of both.

Q15. What does H do on q0 in the circuit?
A15. After CX q0 -> q1, H on q0 turns the Bell basis into the computational basis, so a normal
     measurement (or controlled gates) can read Alice's two bits.

Q16. What are ry and rz?
A16. Rotations on the Bloch sphere around the Y and Z axes. ry(theta) changes the probability of
     0 and 1; rz(phi) changes the phase. Together they can prepare any single-qubit state.

Q17. What is the Bloch sphere?
A17. A picture of one qubit as a point on a sphere: north pole = |0>, south pole = |1>, other
     points are superpositions. Gates are rotations of this sphere.

Q18. What is a Pauli gate?
A18. X (bit flip), Y and Z (phase flip) are the three Pauli gates; with the identity they are
     the basic single-qubit error and correction operations.

Q19. Is the teleported state the same as the original?
A19. Yes. Bob's qubit ends up in exactly Alice's original state (fidelity 1 in a noise-free
     simulation). The original is destroyed.

Q20. What happens on a real quantum computer?
A20. Noise in gates and measurement gives some wrong results, so q2 would not be exactly 1
     every time. The results would show a few percent errors depending on the device.

Q21. Why are the counts split into four groups?
A21. Alice's two measured bits (q0, q1) are random: 00, 01, 10, 11 each with 25 percent. Bob's
     bit is the same in all of them.

Q22. How many qubits and gates are used?
A22. 3 qubits, 3 classical bits. Gates: H, CX, CX, H, CX, CZ plus the message preparation and
     three measurements.

Q23. What is the difference between entanglement and superposition?
A23. Superposition is about a single qubit being in a combination of states. Entanglement is a
     relation between two or more qubits. A Bell pair needs both: H gives the superposition and
     CX makes the entanglement.

Q24. What is no-cloning theorem?
A24. It says an unknown quantum state cannot be copied perfectly. That is why teleportation
     must destroy the original.

Q25. Where is teleportation used?
A25. In quantum networks and repeaters, to connect quantum computers, and inside fault-tolerant
     quantum computing as gate teleportation.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Change the code to teleport |0> instead of |1>. What do you edit?
A1. Remove the X gate: prepare=lambda qc: None. Then q2 should always read 0 in test 1.

Q2. Write the line that creates the Bell pair between q1 and q2.
A2. qc.h(1) and then qc.cx(1, 2)

Q3. What happens if you remove the CZ correction?
A3. Test 1 still looks fine because |1> has no phase problem, but test 2 fails: q2 reads 1 in
    some shots, because the phase is not corrected.

Q4. What happens if you remove the CX correction from q1 to q2?
A4. Bob's bit is flipped half the time, so q2 in test 1 is 1 only about 50 percent of the
    time.

Q5. How do you teleport the state |+> (equal superposition)?
A5. Use prepare=lambda qc: qc.h(0). To check it, use undo=lambda qc: qc.h(2) and see q2 = 0.

Q6. How would you change shots?
A6. Change shots=1024 in the sim.run calls, for example shots=4000.

Q7. How would you measure only Bob's qubit?
A7. Use QuantumCircuit(3, 1) and qc.measure(2, 0). Then counts have only one bit.

Q8. How can you print the measurement result of Bob only in the counts?
A8. Take the leftmost character of each key: bits[0], as done in the code (bob1 and bob2).

Q9. Change the angles in test 2. Does the result change?
A9. No. For any theta and phi the inverse preparation brings q2 back to 0, so Bob still
    measures 0 in all shots.

Q10. Write the line that makes the inverse of ry(theta) on qubit 2.
A10. qc.ry(-theta, 2)

Q11. What changes if you apply the inverse in the wrong order (ry(-theta) first, then
     rz(-phi))?
A11. The state is not undone exactly (the rotations do not commute), so q2 would not always be
     0. The inverse of "ry then rz" is "rz inverse then ry inverse".

Q12. What is deferred measurement and how does the classical-bit version look?
A12. Deferred measurement means: instead of measuring q0, q1 and then applying X or Z on q2
     based on the classical bits, use q1 and q0 directly as quantum controls (cx q1->q2,
     cz q0->q2) and measure at the end. The statistics are identical. The classical-bit
     version in qiskit 2.x (c_if is gone, if_test replaces it) is:
         qc.measure(0, 0); qc.measure(1, 1)
         with qc.if_test((qc.clbits[1], 1)):
             qc.x(2)
         with qc.if_test((qc.clbits[0], 1)):
             qc.z(2)
     Here q1 is measured into c1 and q0 into c0; X is applied if c1 = 1 and Z if c0 = 1.

END OF VIVA NOTES
'''
