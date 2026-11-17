# CL-2 Lab Practicals
## At your own risk. have backups. 
Python code for the two CL-2 lab subjects (Computer Laboratory II, 417526). Every script has run
instructions at the top, step-by-step comments, and theory + viva notes at the end.

All scripts are in the repo root; files are prefixed `QAI_` or `IR_` by subject.
Run any script with `python <file>.py` (or paste it into Jupyter, each file works as one cell).

Install: `pip install qiskit qiskit-aer pgmpy==1.1.2 nltk scikit-learn scipy pandas matplotlib`
(NLTK downloads its small data packages automatically on the first run.)

The quantum scripts run on the local `AerSimulator` from `qiskit-aer`. No IBM account or API token
is needed (the manual's `IBMQ` / `execute` code no longer exists in current qiskit).

## QAI - Quantum AI
| No. | File | Topic |
|---|---|---|
| 1 | `QAI_1_Random_Number_16Qubit.py` | 16-qubit random number generator |
| 2 | `QAI_2_Error_Correction_Noise.py` | Tackle noise with error correction (3-qubit repetition code, readout mitigation) |
| 3 | `QAI_3_Tarrataca_3Puzzle.py` | Tarrataca's quantum production system with the 3-puzzle (Grover over rule sequences) |
| 4 | `QAI_4_Quantum_Teleportation.py` | Quantum teleportation |
| 5 | `QAI_5_QFT_5Qubit.py` | 5-qubit Quantum Fourier Transform and its inverse |

## IR - Information Retrieval
| No. | File | Topic |
|---|---|---|
| 1 | `IR_1_Text_Preprocessing.py` | Text pre-processing: stop words, stemming, lemmatization |
| 2 | `IR_2_Inverted_Index_Retrieval.py` | Document retrieval using an inverted file (Boolean + TF-IDF) |
| 3 | `IR_3_Bayesian_Network_Heart.py` | Bayesian network for heart disease diagnosis |
| 4 | `IR_4_Spam_Filter_Text_Classification.py` | E-mail spam filtering (Naive Bayes + Porter stemming) |
| 5 | `IR_5_Agglomerative_Clustering.py` | Agglomerative hierarchical clustering (credit card customers) |

## Datasets
Datasets live in https://github.com/MrNobody4869/datasets. Every IR script has two options in its load step:

```python
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/<file>.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("<file>.csv")
```

Option A is active by default. To use a local file, comment out Option A and uncomment Option B.

| Script | Dataset file |
|---|---|
| IR_1_Text_Preprocessing.py | `BBC_News.csv` |
| IR_2_Inverted_Index_Retrieval.py | `BBC_News.csv` |
| IR_3_Bayesian_Network_Heart.py | `heart.csv` |
| IR_4_Spam_Filter_Text_Classification.py | `Enron_Spam_5000.csv` |
| IR_5_Agglomerative_Clustering.py | `CC_GENERAL.csv` |

The QAI scripts need no file.

Sources: BBC news articles (Kaggle yufengdev/bbc-fulltext-and-category), Cleveland heart disease
(`heart.csv`, Kaggle; 1025 rows with 302 unique, label inverted, the script fixes both),
Enron-Spam 5000-row sample (from Kaggle bayes2003/emails-for-spam-or-ham-classification-enron-2006,
Metsis et al. 2006), credit card customers (Kaggle "Credit Card Dataset for Clustering").
