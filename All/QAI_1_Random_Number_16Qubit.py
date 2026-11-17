"""
Practical 1 : Quantum Random Number Generator (16 qubits)
=========================================================
INSTRUCTIONS
 1. Install:  pip install qiskit qiskit-aer matplotlib numpy
 2. Dataset : None needed. Everything runs on qiskit's local AerSimulator (no IBM account/token).
 3. Run     : python QAI_1_Random_Number_16Qubit.py
 4. Steps   : build 16-qubit circuit -> Hadamard on all -> measure -> run 1 shot -> read 16-bit number
              -> draw circuit -> repeat 1000 times -> check mean and histogram
"""

# --- Step 1: Import the libraries ---
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# --- Step 2: Create the circuit with 16 qubits and 16 classical bits ---
# The manual used IBMQ.enable_account('API token') and
# provider.get_backend('ibmq_qasm_simulator').
# Here the local simulator is used instead.
n = 16
qc = QuantumCircuit(n, n)

# --- Step 3: Put every qubit in superposition and measure ---
# Hadamard turns |0> into an equal mix of 0 and 1.
# Measuring then gives a random bit.
qc.h(range(n))
qc.measure(range(n), range(n))

# --- Step 4: Run once (shots=1) to get one random 16-bit number ---
# no seed, so every run gives a different number
sim = AerSimulator()
result = sim.run(transpile(qc, sim), shots=1).result()
# e.g. '0110101110010001'
bits = list(result.get_counts())[0]
print("Random bitstring :", bits)
print("Random number    :", int(bits, 2), "(range 0 to 65535)")

# --- Step 5: Draw the circuit as text ---
print(qc.draw(output="text", fold=120))

# --- Step 6: Generate 1000 random numbers (1000 shots, memory=True) ---
shots = 1000
# fixed seed only here, so the 1000-number stats are reproducible
sim_fixed = AerSimulator(seed_simulator=42)
compiled = transpile(qc, sim_fixed)
result = sim_fixed.run(compiled, shots=shots, memory=True).result()
memory = result.get_memory()
numbers = np.array([int(b, 2) for b in memory])
print("Generated", len(numbers), "numbers, first five:", numbers[:5])
print("Mean:", round(numbers.mean(), 1), "(ideal 32767.5)", end=" ")
print("| Min:", numbers.min(), "| Max:", numbers.max())

# --- Step 7: Histogram in 16 bins to check the numbers are spread evenly ---
plt.hist(numbers, bins=16, range=(0, 65536), edgecolor="k")
plt.xlabel("Random number (0 to 65535)")
plt.ylabel("Count")
plt.title("1000 numbers from 16 qubits")
plt.show()






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 1 : QUANTUM RANDOM NUMBER GENERATOR (16 QUBITS)
================================================================================

CODE EXPLANATION
----------------
1. Imports: numpy (to turn bitstrings into an array and take the mean), matplotlib (histogram),
   QuantumCircuit and transpile from qiskit, AerSimulator from qiskit_aer.
2. Circuit: QuantumCircuit(16, 16) makes a quantum register of 16 qubits (all start in |0>) and
   a classical register of 16 bits (to store the measurement results).
3. qc.h(range(16)) applies a Hadamard gate to every qubit. Each qubit becomes (|0> + |1>)/sqrt(2),
   so the whole register is an equal superposition of all 2^16 = 65536 bitstrings.
4. qc.measure(range(16), range(16)) measures qubit i and stores the result in classical bit i.
   Measurement collapses each qubit to 0 or 1 with probability 1/2.
5. One shot: sim.run(transpile(qc, sim), shots=1) runs the circuit once. get_counts() returns a
   dictionary such as {'1111100001010111': 1}. int(bits, 2) converts the bitstring to a number.
6. qc.draw(output="text") prints the circuit as ASCII art (H gate then measurement on each wire).
7. 1000 numbers: shots=1000 with memory=True keeps the result of EVERY shot in order
   (get_memory()). Each one is converted to an integer; mean, min and max are printed.
8. Histogram: plt.hist with 16 bins over 0..65536 shows the numbers are spread evenly.
9. The single number is made WITHOUT a seed, so it differs on every run. Only the 1000-number
   check uses seed_simulator=42 (sim_fixed) so its statistics repeat (see notes on
   pseudo-randomness below). The manual used IBMQ.enable_account('API token') with
   provider.get_backend('ibmq_qasm_simulator'). That is removed in modern qiskit and needs an
   IBM account, so AerSimulator is used locally. To run on
   real hardware you would use the qiskit-ibm-runtime package (QiskitRuntimeService and a
   backend) instead of AerSimulator; the circuit stays the same.

EXPECTED OUTPUT (real run, AerSimulator; 1000-number check uses seed_simulator=42)
----------------------------------------------------------------------
   Random bitstring : 1001010000011111
   Random number    : 37919 (range 0 to 65535)
   (The single number is unseeded, so it will be DIFFERENT on every run; the two lines above
    are just one real example.)
   (circuit drawing: 16 wires, each with an H box then an M box, classical register c: 16)
   Generated 1000 numbers, first five: [ 7583  2168 45929 25738 56110]
   Mean: 32522.0 (ideal 32767.5) | Min: 87 | Max: 65515
   (The histogram window shows 16 bars, each about 1000/16 = 62 high, with small random
   differences. The 1000 numbers use a fixed seed, so you get the same ones on the same qiskit-aer version;
   on another version the numbers may differ but the mean stays near 32768.)

OUTPUT GRAPHS EXPLAINED (and circuit diagrams for quantum scripts)
------------------------------------------------------------------
Each item: what it shows, how to read it, our result, a viva answer, a common mistake.
A. CIRCUIT DIAGRAM (text drawing in the console)
   - Plots: the 16-qubit circuit. Rows q_0 to q_15 are qubits; time runs left to right.
   - The bottom row "c: 16/" is the 16-bit classical register that stores the results.
   - [ H ] = Hadamard gate: puts a qubit in a 50/50 mix of 0 and 1.
   - [M] = measurement; its double line goes to classical bit 0 to 15 (the small number).
   - Read it: every wire is H then M; no gate joins wires, so qubits are independent.
   - Ours: bitstring 1001010000011111 = number 37919 (unseeded, so it changes every run).
   - If asked: "H on 16 qubits gives 2^16 = 65536 equal outcomes; we measure one."
   - Misreading: the staircase of M boxes is only layout, not a chain.

B. FIGURE 1 - HISTOGRAM "1000 numbers from 16 qubits"
   - Plots: how our 1000 random numbers spread over 0 to 65535, in 16 equal bins (bars).
   - x axis: the random number (0 to 65535); each bin is 4096 wide.
   - y axis: Count = how many of the 1000 numbers fall in that bin. No legend (one data set).
   - Read it: a tall bar means many numbers in that range; a fair generator gives similar bars.
   - The ideal bar is 1000/16 = 62.5, so small ups and downs are just chance.
   - Ours: bars about 50 to 78; mean 32522.0 (ideal 32767.5), min 87, max 65515.
   - If asked: "Bars are about equal, so every number is equally likely."
   - Misreading: bars need not be exactly equal and the shape is not a bell curve.

DETAILED THEORY NOTES
---------------------
1. WHAT IS A QUANTUM RANDOM NUMBER GENERATOR?
   - Definition: a device or program that produces random bits by measuring a quantum system
     that is in superposition.
   - Type: quantum circuit (gate model), simplest possible quantum algorithm. No learning.
   - Intuition: a qubit in the state |+> = (|0> + |1>)/sqrt(2) is like a perfectly fair coin
     that has not been tossed yet. Measuring it is the toss. Do this for 16 qubits and you
     get 16 fair coin tosses = a 16-bit number.
   - Maths in words: the Hadamard gate H maps |0> to (|0> + |1>)/sqrt(2). By the Born rule the
     probability of getting a result is the square of its amplitude: (1/sqrt(2))^2 = 1/2 for
     0 and 1/2 for 1. For n qubits every one of the 2^n bitstrings has probability 1/2^n, so
     every integer from 0 to 2^n - 1 is equally likely. For n = 16 that is 1/65536 each.
   - Why it is special: in a normal computer random numbers are pseudo-random (computed by a
     formula from a seed, so predictable if you know the seed). On REAL quantum hardware the
     result of a measurement is fundamentally unpredictable, so it is TRUE randomness.
   - Important honesty point: AerSimulator is a classical program. It uses a pseudo-random
     generator inside, so on the simulator the numbers are only pseudo-random (and repeat if
     the seed is the same). True randomness needs a real quantum device.

2. TYPES AND VARIANTS
   - Pseudo-random generators (PRNG) : Mersenne Twister, LCG. Fast, repeatable, not secure.
   - Cryptographic PRNG (CSPRNG) : OS entropy based, used for keys and passwords.
   - Hardware (classical physical) RNG : electrical noise, radioactive decay, thermal noise.
   - Quantum RNG : photon beam splitters, vacuum fluctuations, or qubit measurement as here.
   - Qiskit variants : change the number of qubits (n) for an n-bit number; use shots to get
     many numbers at once; use only some qubits to get a smaller range (e.g. 3 qubits = 0..7).

3. KEY CONCEPTS AND TERMS
   - Qubit : the quantum bit. State a|0> + b|1>, with a, b complex numbers and |a|^2 + |b|^2 = 1.
   - Superposition : a qubit being in a combination of 0 and 1 at the same time.
   - Measurement : reading a qubit. It collapses to 0 or 1 randomly according to |a|^2, |b|^2.
   - Quantum register : a group of qubits (QuantumRegister). Classical register : a group of
     ordinary bits that stores measurement results (ClassicalRegister).
   - Quantum circuit : an ordered list of gates and measurements applied to registers.
   - Hadamard gate (H) : single qubit gate that creates an equal superposition.
   - Shot : one run of the circuit from start to measurement. shots=1000 = 1000 repeats.
   - Counts : dictionary {bitstring: number of times seen}.
   - memory=True : also keep the shot-by-shot list of results (in order), not just counts.
   - Backend : the thing that runs the circuit (simulator or real quantum computer).
   - Qiskit bit order : the bitstring is written with qubit 15 on the LEFT and qubit 0 on the
     RIGHT (little endian). It does not matter for random numbers but matters for other tasks.
   - Transpile : rewrites the circuit into gates the backend understands.

4. HOW TO JUDGE THE RESULT
   - Mean of many numbers should be near (2^16 - 1)/2 = 32767.5. We got 32522.0 for 1000 numbers.
   - Histogram should be flat (all bins similar). Small differences are normal; for 1000
     samples in 16 bins each bin is about 62 +- 8.
   - Min should be near 0 and max near 65535 (we got 87 and 65515).
   - Formal tests (not in this code): chi-square test for uniformity, NIST SP 800-22 test suite,
     runs test for independence.
   - Standard deviation check: for a uniform number on 0..65535 the std is about 18919.

5. DATA SPLIT / RELEVANT SETUP
   - No data and no train/test split. The only settings are n (number of qubits = number of
     bits), shots (how many numbers) and the seed (to repeat results).
   - Bigger n gives bigger range: n qubits give numbers from 0 to 2^n - 1.

6. THE SIMULATOR (AerSimulator)
   - qiskit_aer.AerSimulator is a fast classical simulator of quantum circuits. It stores
     amplitudes of the state; 16 qubits = 65536 amplitudes, which is easy for a laptop.
     (Each extra qubit doubles memory, so about 30 qubits is the practical limit.)
   - It is noise free unless you give it a noise model. Real devices have noise.
   - seed_simulator=42 fixes its internal random generator so results can be repeated (used
     only for the 1000-number check; the single number has no seed).
   - No account or API token is required. The old manual code used IBMQ and execute(), both
     removed from current qiskit. Modern form is sim.run(transpile(qc, sim), shots=N).

7. ADVANTAGES
   - Very short code (one H gate per qubit and measurement).
   - On real hardware gives genuinely unpredictable numbers, useful for cryptography.
   - Easy to scale: more qubits, more bits.
   - A good first example to learn superposition and measurement.

8. DISADVANTAGES
   - On a simulator it is only pseudo-random, so it gives no real benefit over numpy.random.
   - Real quantum computers are slow to access (queues), cost money or need credits, and have
     noise and bias (the 0/1 probabilities are not exactly 50-50), so post-processing
     (randomness extraction) is needed.
   - Dedicated hardware QRNGs (photon based) are cheaper and faster than using a full quantum
     computer just for random bits.

9. REAL-WORLD USES
   - Cryptographic key generation and one-time pads.
   - Secure communication (quantum key distribution needs random basis choices).
   - Lotteries, gaming and gambling systems.
   - Monte Carlo simulations and scientific sampling.
   - Random seeds for machine learning (weight initialisation, shuffling).

10. COMPARISON WITH RELATED METHODS
   - Quantum RNG vs Python random module : random uses Mersenne Twister (deterministic,
     predictable from its state); QRNG on real hardware is not predictable.
   - Quantum RNG vs os.urandom / secrets : those use OS entropy and are good for security;
     QRNG is a different physical source of entropy.
   - Quantum RNG vs hardware noise RNG : both physical; the quantum one has a theoretical
     guarantee of unpredictability, the classical one depends on chaotic but in principle
     deterministic physics.
   - Simulator vs real device : same circuit, but only the real device gives true randomness.

LIBRARIES AND FUNCTIONS USED
----------------------------
- numpy (np) : np.array makes an array from the list of integers; .mean(), .min(), .max() give
  the summary numbers.
- matplotlib.pyplot (plt) : hist (histogram; bins=16, range=(0, 65536), edgecolor), xlabel,
  ylabel, title, show.
- qiskit.QuantumCircuit(n, n) : circuit with n qubits and n classical bits.
- qc.h(range(n)) : Hadamard on every listed qubit.
- qc.measure(qubits, clbits) : measure each qubit into the matching classical bit.
- qc.draw(output="text", fold=120) : ASCII drawing; fold=120 wraps lines at 120 characters.
- qiskit.transpile(qc, sim) : converts the circuit to the simulator's basis gates.
- qiskit_aer.AerSimulator(seed_simulator=42) : local simulator with a fixed seed (without the
  argument the seed is random, as for the single number).
- sim.run(circuit, shots=..., memory=...) : starts the job; .result() waits for it.
- result.get_counts() : dictionary of bitstring -> count. result.get_memory() : list of
  bitstrings, one per shot (needs memory=True).
- int(bits, 2) : converts a binary string to an integer.

DETAILED VIVA Q&A
-----------------
Q1. What is Qubit?
A1. A qubit is the basic unit of quantum information. Unlike a bit that is only 0 or 1, a
    qubit can be in a superposition a|0> + b|1> where a and b are complex amplitudes and
    |a|^2 + |b|^2 = 1. When measured it gives 0 with probability |a|^2 and 1 with
    probability |b|^2.

Q2. What are different steps performed to generate random number?
A2. (1) Import libraries and create a quantum circuit with n qubits and n classical bits.
    (2) Apply a Hadamard gate on every qubit to create superposition. (3) Measure all qubits
    into the classical bits. (4) Run the circuit on a simulator or real backend with shots=1.
    (5) Read the bitstring from the counts. (6) Convert the bitstring to an integer.

Q3. What are different types of registers?
A3. There are two: the quantum register (QuantumRegister), which holds qubits, and the
    classical register (ClassicalRegister), which holds ordinary bits used to store
    measurement results. QuantumCircuit(16, 16) creates one of each automatically.

Q4. What is quantum circuit?
A4. A quantum circuit is a sequence of quantum gates, measurements and resets applied to
    qubits, drawn as wires from left to right (time goes left to right). It is the quantum
    equivalent of a program or of a logic circuit.

Q5. How you measure the qubits?
A5. With qc.measure(qubit_list, classical_bit_list). Each qubit collapses to 0 or 1 and the
    result is saved in the matching classical bit. Running the circuit with shots=N repeats
    the measurement N times, and get_counts() shows how many times each result appeared.

Q6. What is Hadamard gate?
A6. A single-qubit gate that creates an equal superposition. H|0> = (|0> + |1>)/sqrt(2) and
    H|1> = (|0> - |1>)/sqrt(2). Its matrix is (1/sqrt(2)) [[1, 1], [1, -1]]. Applying H twice
    returns the original state.

Q7. Why does Hadamard on all 16 qubits give a uniform random number?
A7. Each qubit gives 0 or 1 with probability 1/2, independently of the others. So each of the
    65536 bitstrings has probability (1/2)^16, which is the same for all. That is a uniform
    distribution over 0..65535.

Q8. How many different numbers can 16 qubits give?
A8. 2^16 = 65536 numbers, from 0 (0000000000000000) to 65535 (1111111111111111).

Q9. Is the output of this program truly random?
A9. On a real quantum computer, yes (the randomness comes from quantum measurement). On
    AerSimulator, no: it is a classical program using a pseudo-random generator, and with the
    same seed it repeats the same numbers. We use it because it is free and needs no account.

Q10. What is the Born rule?
A10. The probability of measuring a basis state is the square of the magnitude of its
     amplitude. For H|0>, the amplitude of both 0 and 1 is 1/sqrt(2), so each has probability
     1/2.

Q11. What is a shot?
A11. One execution of the circuit from start to measurement. Because quantum results are
     probabilistic, we repeat many shots to see the distribution. Here shots=1 gives one
     number and shots=1000 gives 1000 numbers.

Q12. What does memory=True do?
A12. It tells the backend to store the result of every individual shot in order. get_memory()
     then returns a list of 1000 bitstrings. Without it we only get counts (how many times each
     bitstring appeared), and the order is lost. With 16 bits almost every bitstring appears
     once, so counts are not useful here.

Q13. Why do we transpile the circuit?
A13. Backends support only some basic gates. transpile rewrites the circuit into those gates
     (and maps qubits to the device layout). Modern qiskit requires it before sim.run on most
     circuits.

Q14. What is the difference between pseudo-random and true random numbers?
A14. Pseudo-random numbers come from a deterministic formula and a seed; if you know the seed
     the whole sequence can be predicted. True random numbers come from a physical process
     that cannot be predicted, such as quantum measurement.

Q15. What does the mean of 1000 numbers tell you?
A15. For a uniform distribution on 0..65535 the expected mean is 32767.5. Our mean 32522.0 is
     close. This is a quick sanity check that the numbers are not biased towards small or
     large values.

Q16. Why is the histogram drawn with 16 bins?
A16. 65536 possible values are grouped into 16 equal ranges of 4096. For uniform random numbers
     each range gets about 1000/16 = 62 numbers, so the bars should look about equally tall.

Q17. What is superposition?
A17. A qubit can be in a combination of the 0 and 1 states at the same time. Only on
     measurement does it become one definite value.

Q18. What is the difference between qubit and classical bit?
A18. A bit is 0 or 1. A qubit can be 0, 1 or a superposition, can be entangled with other
     qubits, and its measurement outcome is probabilistic. n qubits hold 2^n amplitudes.

Q19. Why is the output string read in reverse order of qubits?
A19. Qiskit uses little-endian notation: classical bit 0 (from qubit 0) is the rightmost
     character in the printed bitstring. For random numbers this changes nothing.

Q20. What are the Qiskit changes from the manual code?
A20. execute() and IBMQ.enable_account() are removed. Now we create AerSimulator, call
     transpile, then sim.run(...).result(). Aer is imported from qiskit_aer, not qiskit.

Q21. Which real quantum backends can run this?
A21. IBM Quantum devices through qiskit-ibm-runtime (needs an IBM account and token). The
     circuit is the same; only the backend object changes. Results from real devices include
     noise, so the bits may be slightly biased.

Q22. Can you get a number from 0 to 7 only?
A22. Yes, use 3 qubits: QuantumCircuit(3, 3), h on all, measure, shots=1. 2^3 = 8 values.

Q23. How can noise on a real device affect the random numbers?
A23. Gate and readout errors can make 0 and 1 not exactly equally likely (e.g. 51/49). The
     numbers are still unpredictable but slightly biased, so post-processing (a randomness
     extractor such as von Neumann or hashing) is used for security uses.

Q24. How many gates does the circuit have?
A24. 16 Hadamard gates and 16 measurements. Depth is small: all H gates run in parallel in
     one layer, then all measurements.

Q25. Why use quantum random numbers in cryptography?
A25. Keys must be unpredictable. If a PRNG seed leaks, all keys can be recomputed. A quantum
     source has no hidden seed, so an attacker cannot predict the next bit.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Change the code to make a 8-bit random number. What do you edit?
A1. Set n = 8. The circuit has 8 qubits and 8 classical bits, and the number is in 0..255. The
    histogram range should become (0, 256).

Q2. Write the line that applies Hadamard to qubit 3 only.
A2. qc.h(3)

Q3. What happens if you forget qc.measure?
A3. There is no classical result to read: get_counts() fails (no counts) because nothing was
    recorded. The qubits stay in superposition until measured.

Q4. What happens if you change shots=1 to shots=5 in Step 4?
A4. get_counts() returns a dictionary with up to 5 different bitstrings, each with count 1.
    list(result.get_counts())[0] would then take only the first one. Use memory=True for all.

Q5. What is the effect of removing seed_simulator=42 from sim_fixed?
A5. The 1000 numbers change every run. With the seed they repeat. This shows the simulator is
    pseudo-random. (The single number already has no seed, so it changes every run.)

Q6. Write the line that turns the bitstring '1010' into a number.
A6. int('1010', 2), which gives 10.

Q7. How would you print only the numbers greater than 60000?
A7. print(numbers[numbers > 60000]) (numpy boolean indexing).

Q8. How would you get the number of qubits and gates from the circuit?
A8. qc.num_qubits gives 16; qc.count_ops() gives something like {'h': 16, 'measure': 16};
    qc.depth() gives the depth.

Q9. How would you draw the circuit as a picture instead of text?
A9. qc.draw(output="mpl") (needs matplotlib and pylatexenc) and then plt.show().

Q10. How would you change the number of random numbers generated?
A10. Change the variable shots in Step 6, for example shots = 5000.

Q11. How would you check the standard deviation?
A11. numbers.std(). For uniform 0..65535 it should be around 18900.

END OF VIVA NOTES
'''
