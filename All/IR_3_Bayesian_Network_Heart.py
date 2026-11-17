"""
Practical IR-3 : Bayesian Network for Heart Disease Diagnosis (pgmpy)
=====================================================================
INSTRUCTIONS
 1. Install:  pip install numpy pandas matplotlib scikit-learn pgmpy==1.1.2 networkx
              (1.1.2 is the tested version; the DiscreteBayesianEstimator import
              path is version specific)
 2. Dataset : heart.csv (the manual's Cleveland heart disease file: 1025 rows but
              only 302 unique, 13 features + target 0/1 where 1 = NO disease in
              this file). Option A reads it online from the datasets repo;
              Option B reads a local copy - see the two options in Step 2.
 3. Run     : python IR_3_Bayesian_Network_Heart.py
 4. Steps   : load -> discretise numeric columns -> 80/20 split -> define network
              (DAG) -> learn CPDs -> check model -> inference with Variable
              Elimination (3 demo patients) -> test-set accuracy + confusion
              matrix -> draw the network
"""

# --- Step 1: Import the libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination

# --- Step 2: Load the dataset ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/heart.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Download heart.csv from https://github.com/MrNobody4869/datasets
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("heart.csv")
print("Data shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))

# The file repeats the same patients about 3.4 times. Identical rows would land in
# both train and test (leakage) and inflate the accuracy, so keep each patient once.
rows_before = len(df)
df = df.drop_duplicates()
print("Rows before / after drop_duplicates:", rows_before, "->", len(df))

# In this file target = 1 means NO disease (inverted). Flip it so 1 means disease.
df["condition"] = 1 - df["target"]
df = df.drop(columns="target")
counts = df["condition"].value_counts()
print(counts.rename({0: "0 = healthy", 1: "1 = heart disease"}))

# Evidence for the flip: the disease group has lower max heart rate, more angina.
print(df.groupby("condition")[["thalach", "exang"]].mean().round(2))

# --- Step 3: Discretise the continuous columns into 3 bins each ---
# pgmpy tables need few states, so age, trestbps, chol, thalach become low/mid/high.
# (equal-size bins)
for col in ["age", "trestbps", "chol", "thalach"]:
    df[col] = pd.qcut(df[col], 3, labels=["low", "mid", "high"]).astype(str)
# oldpeak has many zeros, so qcut fails; use fixed cut points instead.
cut_points = [-0.1, 0.5, 2.0, 10]
df["oldpeak"] = pd.cut(df["oldpeak"], cut_points, labels=["low", "mid", "high"])
df["oldpeak"] = df["oldpeak"].astype(str)
print(df.head(3))

# --- Step 4: Train / test split (80 / 20, keep the same disease ratio in both) ---
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["condition"]
)
print("Train rows:", len(train), "| Test rows:", len(test))

# --- Step 5: Define the network structure (the DAG) ---
# Risk factors point to condition; condition points to its symptoms and test results.
edges = [("age", "condition"), ("sex", "condition"), ("chol", "condition"),
         ("trestbps", "condition"), ("fbs", "condition"),
         ("condition", "cp"), ("condition", "restecg"), ("condition", "thalach"),
         ("condition", "exang"), ("condition", "oldpeak"), ("condition", "slope"),
         ("condition", "ca"), ("condition", "thal")]
model = DiscreteBayesianNetwork(edges)

# --- Step 6: Learn the CPDs from the training data ---
# DiscreteBayesianEstimator (BDeu prior) adds small pseudo-counts so no probability
# is exactly 0.
# state_names lists every possible value of each column, so unseen values never
# crash later.
state_names = {}
for c in df.columns:
    state_names[c] = sorted(df[c].unique())
estimator = DiscreteBayesianEstimator(
    state_names=state_names, prior_type="BDeu", equivalent_sample_size=10
)
model.fit(train, estimator=estimator)
print("\nCPD of cp (chest pain) given condition:")
print(model.get_cpds("cp"))
print("CPD of exang (exercise angina) given condition:")
print(model.get_cpds("exang"))

# --- Step 7: Check that the model is valid ---
# Checks the graph is a DAG and every CPD column sums to 1.
print("Model valid:", model.check_model())

# --- Step 8: Inference with Variable Elimination - 3 demo patients ---
infer = VariableElimination(model)
# Demo input: hand-typed evidence for three patients (only some facts are known).
# Codes of THIS file: cp 0 = asymptomatic, 3 = typical angina; thal 3 = reversible.
patients = {
    "Patient A (old male, asymptomatic chest pain cp 0, exercise angina)":
        {"age": "high", "sex": 1, "cp": 0, "exang": 1},
    ("Patient B (young female, typical-angina chest pain cp 3, "
     "no exercise angina, high max heart rate)"):
        {"age": "low", "sex": 0, "cp": 3, "exang": 0, "thalach": "high"},
    ("Patient C (middle age male, 2 blocked vessels, "
     "thal 3 = reversible defect, high ST depression)"):
        {"age": "mid", "sex": 1, "ca": 2, "thal": 3, "oldpeak": "high"},
}
for name, evidence in patients.items():
    q = infer.query(["condition"], evidence=evidence, show_progress=False)
    print(name, "->", evidence)
    print("   P(condition=1 | evidence) =", round(float(q.values[1]), 3))

# --- Step 9: Evaluate on the test set ---
# For every test row, give all other columns as evidence, pick the likelier condition.
y_true = []
y_pred = []
for _, row in test.iterrows():
    evidence = row.drop("condition").to_dict()
    q = infer.query(["condition"], evidence=evidence, show_progress=False)
    y_pred.append(int(q.values[1] > 0.5))
    y_true.append(int(row["condition"]))
print("\nTest accuracy :", round(accuracy_score(y_true, y_pred), 3))
print("Precision     :", round(precision_score(y_true, y_pred), 3))
print("Recall        :", round(recall_score(y_true, y_pred), 3))
print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))

# --- Step 10: Draw the network graph ---
# Convert the edge list to a networkx graph. shell_layout puts "condition" in the
# centre and all other nodes on a circle around it, so every arrow can be read easily.
g = nx.DiGraph(edges)
other_nodes = []
for n in g.nodes:
    if n != "condition":
        other_nodes.append(n)
pos = nx.shell_layout(g, nlist=[["condition"], other_nodes])
nx.draw(g, pos, with_labels=True, node_color="lightblue",
        node_size=1800, font_size=8, arrowsize=15)
plt.title("Bayesian network for heart disease")
plt.show()






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL IR-3 : BAYESIAN NETWORK FOR HEART DISEASE
================================================================================

CODE EXPLANATION
----------------
1. Imports: pandas for data, networkx + matplotlib to draw the graph, sklearn for the split
   and the scores, pgmpy for the network (DiscreteBayesianNetwork), the estimator
   (DiscreteBayesianEstimator) and the inference engine (VariableElimination).
2. Load data: Option A (active) reads heart.csv from the datasets repo; Option B (commented)
   reads a local copy. The file has 1025 rows, 13 features and a 0/1 column "target". No
   missing values. The same 302 patients are repeated about 3.4 times, so drop_duplicates()
   keeps 302 rows (otherwise identical rows would fall in both train and test and make the
   accuracy look better than it is). In this file target = 1 means NO disease, so we create
   condition = 1 - target (1 = disease) and drop target. The class counts (138 disease /
   164 healthy) and the mean thalach / exang per condition are printed to prove the
   orientation: the disease group has the lower max heart rate and more exercise angina.
3. Discretise: pgmpy's CPDs are tables, so continuous columns would give huge tables. age,
   trestbps, chol and thalach are cut into 3 equal-sized bins (low / mid / high) with
   pd.qcut. oldpeak has many zeros so qcut cannot make equal bins; pd.cut with fixed edges
   (0.5 and 2.0) is used. The other columns are already small categories and stay as they are.
4. Split: 80 / 20 with random_state=42 and stratify=condition, so train and test have the
   same healthy / sick ratio.
5. Structure (DAG): risk factors age, sex, chol, trestbps, fbs point TO condition; condition
   points TO the symptoms and test results cp, restecg, thalach, exang, oldpeak, slope, ca,
   thal. This follows the idea "disease causes symptoms; risk factors cause disease".
6. Learn CPDs: model.fit(train, estimator=DiscreteBayesianEstimator(...)) counts how
   often each value occurs for every parent combination and adds small pseudo-counts so no
   probability is exactly 0. state_names tells pgmpy all possible values of every column.
7. check_model(): confirms the graph is a DAG and every CPD column adds up to 1.
8. Inference: VariableElimination(model).query(["condition"], evidence=...) gives
   P(condition | evidence) for a patient. Three hand-typed demo patients are diagnosed. Their
   evidence uses THIS file's codes (cp 0 = asymptomatic, cp 3 = typical angina, thal 3 =
   reversible defect), which differ from the UCI documentation (see theory section 6).
9. Evaluation: for each test row all other columns are used as evidence; if
   P(condition=1) > 0.5 we predict "disease". Accuracy, precision, recall and the confusion
   matrix are printed.
10. Graph: the edge list is converted to a networkx DiGraph. nx.shell_layout puts "condition"
    in the centre and the other 13 nodes on a circle around it (arrows come in from the risk
    factors and go out to the symptoms), which gives a clear star-shaped picture.

EXPECTED OUTPUT (real run, Option A data, seed 42)
----------------------------------------------------------------------
   Data shape: (1025, 14)
   Missing values: 0
   Rows before / after drop_duplicates: 1025 -> 302
   Class counts: 0 = healthy 164, 1 = heart disease 138
   Mean per condition: thalach 158.38 (healthy) vs 139.1 (disease); exang 0.14 (healthy) vs 0.55 (disease)
   Train rows: 241 | Test rows: 61
   CPD of cp given condition (columns condition(0), condition(1)):
      cp(0) 0.222 0.715 | cp(1) 0.267 0.080 | cp(2) 0.406 0.150 | cp(3) 0.105 0.054
      (in this file cp 0 = asymptomatic, so disease patients are mostly cp 0)
   CPD of exang given condition: exang(1) = 0.136 (healthy) and 0.561 (disease)
   Model valid: True
   Patient A (age high, sex 1, cp 0, exang 1)              -> P(condition=1 | evidence) = 0.978
   Patient B (age low, sex 0, cp 3, exang 0, thalach high) -> P(condition=1 | evidence) = 0.02
   Patient C (age mid, sex 1, ca 2, thal 3, oldpeak high)  -> P(condition=1 | evidence) = 0.993
   Test accuracy : 0.82
   Precision     : 0.815
   Recall        : 0.786
   Confusion matrix [[TN FP] [FN TP]] = [[28 5] [6 22]]
   With only 61 test rows, one patient moves the accuracy by about 1.6 points, so the accuracy
   has wide error bars (roughly plus or minus 10 points); do not read much into 0.82 versus 0.85.
   (A network picture opens at the end. A pgmpy FutureWarning about StructureScore may print
   on import; it comes from inside pgmpy and is harmless.)

OUTPUT GRAPHS EXPLAINED
-----------------------
One figure only: the network graph (Figure 1, Step 10).

GRAPH 1 : BAYESIAN NETWORK GRAPH (title "Bayesian network for heart disease")
- Plots: the structure of the model (the edge list of Step 5), not data values.
- Nodes (light blue circles): the 14 variables. "condition" is in the centre, 13 around it.
- Arrow A -> B: A is a parent of B, so A directly influences B. There are no axes.
- 5 arrows come IN to condition (risk factors) and 8 go OUT (symptoms and test results).
- Read it as cause to effect: risk factors -> disease -> symptoms. There is no loop.
- OUR result: model valid (check_model True); P(disease) = 0.978 / 0.02 / 0.993, accuracy 0.82.
- If asked: "Arrows go from cause to effect, and the CPTs on the nodes let us infer disease."
- Misreading: position and arrow length mean nothing; the numbers are in the CPDs.

DETAILED THEORY NOTES
---------------------
1. WHAT IS A BAYESIAN NETWORK?
   - Definition: a Bayesian network (belief network) is a probabilistic graphical model made of
     (a) a DAG: nodes = random variables, directed edges = direct influence (dependence), and
     (b) a CPT (conditional probability table) for every node, giving P(node | its parents).
   - Type: here it is used like a SUPERVISED model (we know condition in the training data),
     but the same network can answer any question, not only "condition".
   - Intuition: like a flowchart of causes and effects with probabilities on the arrows.
     Electricity failure and computer malfunction both cause a computer failure (manual's
     example). Seeing the effect lets us reason backwards to the likely cause.
   - Maths in words: the joint probability of all variables is the product of each variable's
     probability given its parents: P(X1..Xn) = product of P(Xi | parents(Xi)). This is the
     chain rule plus the conditional independence the graph claims. Bayes theorem is then used
     to reverse the direction: P(cause | evidence) = P(evidence | cause) * P(cause) / P(evidence).

2. TYPES AND VARIANTS
   - Discrete Bayesian network : every variable has a finite set of values (this practical).
   - Gaussian / linear Gaussian network : continuous variables with normal distributions.
   - Naive Bayes : the simplest network, one class node pointing to all features.
   - Dynamic Bayesian network : variables repeated over time (speech, tracking; an HMM is a case).
   - Structure can be given by an expert (we did this) or learned from data (hill climbing,
     PC algorithm, score based search).
   - Parameters can be learned by Maximum Likelihood (plain counts) or Bayesian estimation
     (counts plus prior).

3. KEY CONCEPTS AND TERMS
   - DAG : directed acyclic graph - arrows with no loop back to a start node.
   - Parent / child : A -> B means A is a parent of B.
   - CPT / CPD : conditional probability table / distribution P(node | parents). Each column
     sums to 1.
   - Evidence : the variables whose values we know (age, sex, chest pain ...).
   - Query variable : the variable we want to know (condition).
   - Conditional independence : A and B are independent given C if once C is known, B gives no
     extra information about A. It is what keeps the tables small.
   - d-separation : a graph rule to read conditional independence. A path is blocked if it has
     a chain A->C->B or fork A<-C->B with C known, or a collider A->C<-B with C (and its
     descendants) NOT known. Example here: given condition, the symptoms cp and thalach are
     independent of each other (fork through condition).
   - Marginalisation : summing out variables we do not care about.
   - Variable Elimination : exact inference. It multiplies the CPT factors and sums out the
     unwanted variables one at a time, instead of building the whole joint table.
   - Laplace / BDeu smoothing : adding pseudo-counts so that an unseen combination gets a small
     non-zero probability instead of 0.
   - Discretisation : turning numbers into a few ranges (low / mid / high).

4. EVALUATION METRICS
   - Accuracy = (TP + TN) / all test rows. Overall correctness.
   - Precision = TP / (TP + FP). Of the patients we called sick, how many really are.
   - Recall (sensitivity) = TP / (TP + FN). Of the really sick patients, how many we found. In
     medicine recall is very important because a missed patient (FN) is dangerous.
   - Confusion matrix : table of TN, FP, FN, TP.
   - The posterior probability itself is also an output: 0.9 is a stronger diagnosis than 0.55.

5. DATA SPLIT / RELEVANT SETUP
   - Duplicates are dropped first (1025 -> 302 rows), then 80 % train (241 rows) / 20 % test
     (61 rows), random_state=42, stratify=condition.
   - CPDs are learned ONLY from train; the test rows are used only for checking.
   - Bin edges for qcut were computed on the whole table before the split (a tiny, harmless
     leak because only 3 coarse bins are used; a stricter version would compute them on train).

6. THE DATASET
   - File: heart.csv, the manual's Cleveland heart disease data (UCI, 76 attributes but 14 columns
     are used). It has 1025 rows but only 302 UNIQUE patients: the same patients are repeated
     about 3.4 times. If duplicates stay, identical rows land in both train and test, the model
     just "remembers" them and the accuracy is falsely high (data leakage). So we call
     drop_duplicates() and work with 302 rows. The manual describes 303 rows and a 0-4 label;
     this file is the same data with a binary label and 302 unique rows (the missing-value rows
     are kept: ca = 4 in 4 rows and thal = 0 in 2 rows are the original '?' entries, treated
     here as just another category).
   - Label: the column is called target and in this file target = 1 means NO heart disease and
     target = 0 means disease (the usual coding is the other way round). We know because after
     removing duplicates the target = 1 group has mean thalach 158.4, oldpeak 0.59, exang 0.14,
     ca 0.34, while the target = 0 group has thalach 139.1, oldpeak 1.59, exang 0.55, ca 1.17;
     sick patients have lower max heart rate and more angina, ST depression and blocked vessels.
     The same patients in the other Cleveland file (label condition, 1 = disease) confirmed it:
     matched patient by patient, target 0 always equals condition 1. Fix: condition = 1 - target,
     so condition 1 = disease. Class counts after the fix: 138 disease / 164 healthy (fairly
     balanced).
   - Columns: age (years); sex (1 male, 0 female); cp chest pain type; trestbps resting blood
     pressure; chol serum cholesterol; fbs fasting blood sugar > 120 mg/dl; restecg resting ECG
     result; thalach maximum heart rate; exang exercise induced angina; oldpeak ST depression;
     slope of the ST segment; ca number of major vessels coloured by fluoroscopy; thal
     thalassemia test; condition = heart disease present (made by us).
   - CODE TABLES FOR THIS FILE (verified by matching the same patients with the UCI-coded
     Cleveland file, about 296 patients matched one to one):
       cp     : 0 = asymptomatic, 1 = atypical angina, 2 = non-anginal pain, 3 = typical angina
       restecg: 0 = left ventricular hypertrophy, 1 = normal, 2 = ST-T abnormality
       slope  : 0 = downsloping, 1 = flat, 2 = upsloping
       thal   : 0 = missing (2 rows), 1 = fixed defect, 2 = normal, 3 = reversible defect
       ca     : 0-3 vessels (same as UCI), 4 = missing (4 rows)
       sex    : 1 male, 0 female; fbs, exang: 1 = yes, 0 = no
   - WARNING: these cp, restecg, slope and thal codes DIFFER from the UCI documentation (UCI
     cp: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic; this file reverses the order, so
     cp 0 and cp 3 are swapped; slope, restecg and thal also use a different order). Do not
     trust the usual code lists for this file. That is why the demo patients use cp 0 for
     asymptomatic pain, cp 3 for typical angina and thal 3 for a reversible defect.

7. ADVANTAGES
   - Easy to read: the graph shows what depends on what.
   - Handles missing evidence naturally: just leave that variable out of the evidence.
   - Combines expert knowledge (structure) with data (probabilities).
   - Gives probabilities, not only a yes / no.
   - Works with little data compared with deep models, and can explain its decision.

8. DISADVANTAGES
   - Structure must be given by an expert or learned (hard, many possible DAGs).
   - Exact inference is slow for big, densely connected networks (NP-hard in general).
   - CPT size grows fast with the number of parents, so continuous data must be discretised
     and information is lost.
   - Wrong structure (wrong independence assumptions) gives wrong answers.
   - No cycles allowed, so feedback loops cannot be shown.

9. REAL-WORLD USES
   - Medical diagnosis (this practical), risk prediction.
   - Spam filtering (Naive Bayes), fault diagnosis in machines, network troubleshooting.
   - Genetics and bioinformatics, speech and gesture recognition.
   - Decision support and risk analysis in finance and insurance.

10. COMPARISON WITH RELATED METHODS
   - Bayesian network vs Naive Bayes : Naive Bayes assumes all features are independent given
     the class; a Bayesian network allows richer links between features.
   - Bayesian network vs logistic regression : logistic regression is a direct discriminative
     model; a Bayesian network is generative and can answer many different queries and handle
     missing inputs.
   - Bayesian network vs decision tree : a tree gives rules; a network gives probabilities and
     reasoning in both directions (cause to effect and effect to cause).
   - Bayesian network vs neural network : neural nets are usually more accurate on big data but
     are black boxes; Bayesian networks are transparent.

LIBRARIES AND FUNCTIONS USED
----------------------------
- pandas (pd) : read_csv loads the file; qcut (equal-size bins), cut (fixed bin edges);
  astype(str) makes the bin labels plain text; value_counts, head, isna().sum();
  drop_duplicates (keep each unique row once); drop(columns=...); groupby(...).mean().
- matplotlib.pyplot (plt) : title, show - shows the network picture.
- networkx (nx) : DiGraph builds a directed graph from the edge list; shell_layout
  (nlist = [["condition"], other nodes]) puts condition in the centre and the rest on a circle;
  nx.draw draws it.
- sklearn.model_selection.train_test_split : test_size=0.2, random_state=42, stratify.
- sklearn.metrics : accuracy_score, precision_score, recall_score, confusion_matrix.
- pgmpy.models.DiscreteBayesianNetwork : the network class (older pgmpy called it
  BayesianModel / BayesianNetwork, as in the manual). Takes the list of edges.
- pgmpy.parameter_estimator.DiscreteBayesianEstimator : learns CPDs with a prior (older pgmpy
  called it estimators.BayesianEstimator). prior_type="BDeu" with equivalent_sample_size=10 is
  the smoothing strength; state_names lists every possible value. (DiscreteMLE, formerly
  MaximumLikelihoodEstimator, is the plain-count version used in the manual.)
- model.fit(data, estimator=...) : learns and attaches all CPDs.
- model.get_cpds(node) : returns the CPD of one node. model.check_model() : validity check.
- pgmpy.inference.VariableElimination : exact inference engine; .query(variables, evidence).
  show_progress=False hides the progress bar.
- q.values : the probability array of the query result, index 0 = condition 0, 1 = condition 1.

DETAILED VIVA Q&A
-----------------
Q1. What is Bayesian network model?
A1. It is a probabilistic graphical model: a DAG whose nodes are random variables and edges
    show direct dependence, plus a conditional probability table for each node. It is used to
    find the probability of a cause given the observed evidence.

Q2. What are the attributes used in Heart Disease Data Set?
A2. age, sex, cp (chest pain type), trestbps (resting blood pressure), chol (cholesterol), fbs
    (fasting blood sugar > 120), restecg (resting ECG), thalach (max heart rate), exang
    (exercise induced angina), oldpeak (ST depression), slope (ST slope), ca (major vessels
    coloured), thal (thalassemia) and the target condition (heart disease yes / no).

Q3. What is pgmpy?
A3. pgmpy is a Python library for probabilistic graphical models. It lets us build Bayesian
    networks, learn their structure and CPDs from data and run inference such as Variable
    Elimination.

Q4. What is a DAG and why must it be acyclic?
A4. A directed acyclic graph has arrows with a direction and no path that returns to its start.
    With a cycle the chain-rule product would be circular and the joint probability could not
    be defined.

Q5. What is a CPT?
A5. A conditional probability table lists P(node value | each combination of parent values).
    For a node with no parent it is just the ordinary prior probability.

Q6. How is the joint probability written in a Bayesian network?
A6. As the product of every node's probability given its parents. This is much smaller than a
    full joint table because of the conditional independence in the graph.

Q7. What is conditional independence?
A7. A and B are conditionally independent given C when, once we know C, learning B does not
    change our belief about A. The network structure encodes these statements.

Q8. What is d-separation?
A8. A graph rule to decide if two variables are independent given some evidence. A path is
    blocked by a known middle node in a chain or fork, and by an UNKNOWN middle node in a
    collider. If all paths are blocked the variables are d-separated (independent).

Q9. How does Variable Elimination work?
A9. It keeps the CPTs as factors, sets the evidence, then multiplies the factors that contain
    a variable and sums that variable out, one variable at a time, until only the query
    variable remains. Finally it normalises. It avoids building the huge joint table.

Q10. Why did you discretise age, chol, etc.?
A10. A CPT needs a finite number of values. A continuous column like chol has 152 different
     values, so its CPT would be huge and mostly empty. Three bins (low, mid, high) keep the
     tables small and the counts large enough.

Q11. Why a Bayesian estimator and not Maximum Likelihood?
A11. Maximum likelihood is plain counting; any combination that never appears in the training
     data gets probability 0, and a single 0 can wipe out the whole product at inference time
     (that is why the manual's output shows 0.0000 for a class). The Bayesian estimator adds a small
     prior (pseudo-counts, like Laplace smoothing) so nothing is exactly 0.

Q12. What does model.check_model() check?
A12. That the graph is a DAG, that each node has a CPD with the right parents and that every
     CPD column sums to 1. It returns True if the model is valid.

Q13. Why do you split the data 80 / 20?
A13. To learn the CPDs on one part and test on unseen patients, so the accuracy is honest.
     stratify keeps the same ratio of sick and healthy in both parts.

Q14. Why does the structure have risk factors pointing to condition and condition pointing to
     symptoms?
A14. It matches medical thinking: age, sex, cholesterol, blood pressure and sugar raise the
     risk of disease (cause), and disease then shows up as chest pain, abnormal ECG, low max
     heart rate and so on (effect). We then reason backwards from symptoms to disease.

Q15. What is the meaning of P(condition=1 | evidence)?
A15. The probability, after seeing the evidence, that the patient has heart disease. We call
     the patient sick if it is above 0.5.

Q16. What if some of the patient's data is missing?
A16. Just leave it out of the evidence. Variable Elimination sums over the unknown variable, so
     the network still gives an answer. That is a big advantage over many other models.

Q17. Which metric is most important for a medical diagnosis model and why?
A17. Recall, because a false negative means a sick patient is sent home. Precision also matters
     so that healthy people are not over-treated, so both are reported with the confusion matrix.

Q18. What is the difference between Naive Bayes and a Bayesian network?
A18. Naive Bayes is a Bayesian network with one class node pointing to every feature and no
     links between features (strong independence assumption). A general Bayesian network may
     have any DAG and so models real dependencies.

Q19. Can the structure be learned from data?
A19. Yes, with score based search (HillClimbSearch with BIC or K2 score), constraint based
     (PC algorithm) or hybrid methods. Here we used an expert structure to keep it simple.

Q20. What is the problem with a node having many parents?
A20. The CPT grows exponentially: with p parents each having s values there are s^p columns.
     Condition has 5 parents with 3, 2, 3, 3, 2 values = 108 columns, and some columns have
     few or no training rows, which is why smoothing is needed.

Q21. What is the difference between inference and learning in a Bayesian network?
A21. Learning builds the network (structure and CPTs) from data. Inference uses the finished
     network to calculate probabilities for new evidence.

Q22. What is exact versus approximate inference?
A22. Exact methods (Variable Elimination, junction tree) give the true answer but can be slow
     for big networks. Approximate methods (sampling, e.g. Gibbs, likelihood weighting) are
     faster but give estimates.

Q23. What is a collider and why is it special?
A23. A collider is a node with two arrows pointing into it (A -> C <- B). A and B are
     independent normally, but become dependent once C (or its descendant) is observed
     ("explaining away"). Example: two risk factors both raising condition.

Q24. Why is the 0 / 1 target easier than the manual's 0-4 target?
A24. We only decide "disease or not". The manual's 0 to 4 scale has very few rows in classes 3
     and 4, which makes the CPTs sparse. Merging 1-4 into 1 gives enough data per value.

Q25. How do you read the output of query()?
A25. It prints a small table: one row per value of condition and its probability, summing to 1.
     q.values[1] is the probability of disease.

Q26. How accurate is the model and can it be improved?
A26. About 0.82 on the 61 test rows (see the expected output). With so few test rows the
     accuracy has wide error bars (one patient is about 1.6 points). It can be improved by a
     better structure (learn it from data), more bins or better cut points, more data, or by
     dropping weak features so the CPT of condition gets smaller.

Q27. Why do you drop duplicate rows?
A27. heart.csv has 1025 rows but only 302 unique patients (each repeated about 3.4 times). If
     we keep them, the same patient can be in both train and test, so the test set is not
     really unseen and the accuracy is inflated (data leakage). We dedupe first, then split.

Q28. Why is the target inverted and how do we know?
A28. In this file target = 1 means NO heart disease and 0 means disease. We checked the
     feature means after removing duplicates: the target = 0 group has lower max heart rate
     (139 vs 158) and more exercise angina (0.55 vs 0.14), more ST depression and more
     blocked vessels, which is what sick patients look like. Matching the same patients with
     the other Cleveland file (condition, 1 = disease) agreed. So we use condition = 1 - target.

Q29. How many unique patients are there?
A29. 302 (the manual says 303 for the original Cleveland data; 1025 rows in the file are
     repeats). 138 have heart disease and 164 do not.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Which line builds the network and what does it take?
A1. model = DiscreteBayesianNetwork(edges). It takes a list of (parent, child) tuples.

Q2. Write the line that gives the probability of disease for an older (high age group) male
    with exang=1.
A2. infer.query(["condition"], evidence={"age": "high", "sex": 1, "exang": 1}). Then read
    .values[1].

Q3. Replace the Bayesian estimator by Maximum Likelihood. What happens?
A3. Use estimator=DiscreteMLE() (older name MaximumLikelihoodEstimator) and no prior. Unseen parent combinations get
    undefined / zero probability, so inference may fail or give 0 and the accuracy usually
    drops.

Q4. What happens if you increase equivalent_sample_size from 10 to 100?
A4. The prior gets more weight: probabilities are pulled toward uniform (0.5 / 0.33) and the
    data counts matter less. Too large a value underfits.

Q5. How do you use 5 bins instead of 3 for chol?
A5. pd.qcut(df["chol"], 5, labels=["very low","low","mid","high","very high"]). CPTs get
    larger and sparser.

Q6. How do you add an edge age -> chol?
A6. Add ("age", "chol") to the edges list. chol already points to condition, so this is still
    a DAG (age -> chol -> condition). Never add an edge that closes a loop.

Q7. What happens if you add an edge that creates a cycle?
A7. If the cyclic edge list is passed to the DiscreteBayesianNetwork constructor, networkx
    raises NetworkXError ('Input is not a valid edge list'). The ValueError 'Loops are not
    allowed' comes when the edge is added later with model.add_edge.

Q8. How do you print the CPD of any other node, e.g. slope?
A8. print(model.get_cpds("slope")).

Q9. How do you see the independencies of the model?
A9. model.get_independencies() lists them. model.is_dconnected("cp", "thalach",
    observed=["condition"]) tests one pair (should be False: blocked by the fork).

Q10. Why do we call astype(str) after qcut?
A10. qcut returns a Categorical column; converting to plain strings makes it behave like the
     other state labels in pgmpy and in the evidence dictionary.

Q11. How would you change the decision threshold from 0.5 to 0.4 and why?
A11. y_pred.append(int(q.values[1] > 0.4)). It labels more patients as sick: recall goes up,
     precision goes down - useful when missing a patient is costly.

Q12. How do you save the picture of the network?
A12. Call plt.savefig("network.png") before plt.show().

Q13. How do you remove the repeated patients and which line makes the label right?
A13. df = df.drop_duplicates() (1025 -> 302 rows) and df["condition"] = 1 - df["target"], then
     df = df.drop(columns="target"). After this condition 1 means heart disease.

Q14. Which value of cp means asymptomatic chest pain in this file?
A14. cp = 0 (and typical angina is cp = 3). This differs from the UCI documentation, so the
     codes were checked by matching the same patients with the UCI-coded Cleveland file.

END OF VIVA NOTES
'''
