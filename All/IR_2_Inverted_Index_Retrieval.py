"""
Practical 2 : Inverted Index and Retrieval (Boolean + TF-IDF ranked) on BBC News
================================================================================
INSTRUCTIONS
 1. Install:  pip install pandas nltk
              (the nltk.download lines in Step 1 fetch punkt_tab and stopwords once;
              internet is needed only the first time)
 2. Dataset : BBC_News.csv (2225 articles; columns category, text). The first 500
              articles are the document collection. Option A fetches it online;
              Option B reads a local copy.
 3. Run     : python IR_2_Inverted_Index_Retrieval.py
 4. Steps   : load 500 docs -> preprocess (lower, tokenize, stop words, stem) ->
              build inverted index -> save/load inverted file -> index statistics ->
              Boolean retrieval (AND/OR/NOT) -> TF-IDF ranked retrieval ->
              compare inverted index lookup with linear scan
"""

# --- Step 1: Import the libraries and download the NLTK data (one time) ---
import json
import math
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

for pkg in ["punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

# --- Step 2: Load the dataset and keep the first 500 articles ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/BBC_News.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Download BBC_News.csv from https://github.com/MrNobody4869/datasets
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("BBC_News.csv")
df = df.head(500)          # document id = row index 0..499
N = len(df)
print("Documents in the collection (N):", N)
print(df.category.value_counts().to_dict())

# --- Step 3: Preprocessing function (same steps as Practical 1) ---
# lowercase -> tokenize -> keep letters only -> remove stop words -> Porter stem.
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    kept = []
    for t in tokens:
        if re.fullmatch(r"[a-z]+", t) and t not in stop_words:
            kept.append(t)
    stems = []
    for t in kept:
        stems.append(stemmer.stem(t))
    return stems

example = "The markets were falling and investors are selling shares"
print("Example:", preprocess(example))

# --- Step 4: Build the inverted index ---
# index = {term: {doc_id: term frequency}}
# also store each document's length for the scan test.
index = {}
doc_terms = {}
for doc_id, text in df.text.items():
    terms = preprocess(text)
    doc_terms[doc_id] = terms
    for term in terms:
        if term not in index:
            index[term] = {}
        index[term][doc_id] = index[term].get(doc_id, 0) + 1

# --- Step 4b: Save the index to a file and load it back ---
# This file IS the "inverted file".
# JSON keys must be strings, so doc ids are saved as "22"
# and turned back into int(22) when loading.
with open("inverted_index.json", "w") as f:
    json.dump(index, f)
with open("inverted_index.json") as f:
    saved = json.load(f)
loaded = {}
for term, postings in saved.items():
    loaded[term] = {}
    for d, tf in postings.items():
        loaded[term][int(d)] = tf
same_index = loaded == index
print("Inverted file saved to inverted_index.json;", end=" ")
print("loaded copy equals the built index:", same_index)
assert loaded == index
index = loaded             # all queries below use the index loaded from the file

# --- Step 5: Index statistics ---
# Vocabulary size, total postings and a sample posting list.
print("\nVocabulary size (unique terms):", len(index))
print("Total postings (term-document pairs):", sum(len(p) for p in index.values()))
first8 = list(index["footbal"].items())[:8]
print("Posting list of 'footbal' (doc_id: tf), first 8:", first8)
print("Document frequency (df) of some terms:")
for t in ["market", "footbal", "mobil", "phone", "game", "govern"]:
    print("  %-8s df = %d" % (t, len(index.get(t, {}))))

# --- Step 6: Boolean retrieval with set operations on posting lists ---
# Each term gives a set of doc ids; AND = intersection, OR = union, NOT = difference.
def docs_with(word):
    terms = preprocess(word)      # [] for a stop word or a number
    return set(index.get(terms[0], {})) if terms else set()

def show(ids, limit=5):
    for d in sorted(ids)[:limit]:
        print("   doc %3d [%s] %s" % (d, df.category[d], df.text[d][:100]))

# Demo input: hand-typed Boolean queries.
print("\nBOOLEAN: 'stock AND market'")
r = docs_with("stock") & docs_with("market")
print("  matches:", len(r))
show(r)

print("\nBOOLEAN: 'football OR rugby'")
r = docs_with("football") | docs_with("rugby")
print("  matches:", len(r))
show(r, 3)

print("\nBOOLEAN: 'mobile AND phone AND NOT game'")
r = (docs_with("mobile") & docs_with("phone")) - docs_with("game")
print("  matches:", len(r))
show(r)

# --- Step 7: Ranked retrieval with TF-IDF ---
# score(doc) = sum over query terms of tf * log(N / df); highest score first.
def search(query, top=5):
    scores = {}
    for term in preprocess(query):
        if term in index:
            idf = math.log(N / len(index[term]))
            for doc_id, tf in index[term].items():
                scores[doc_id] = scores.get(doc_id, 0) + tf * idf
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top]

# Demo input: hand-typed ranked queries.
for q in ["stock market", "football match", "mobile phone"]:
    print("\nRANKED QUERY:", q, "->", preprocess(q))
    for doc_id, score in search(q):
        category = df.category[doc_id]
        snippet = df.text[doc_id][:100]
        print("   doc %3d  score %6.2f  [%s] %s" % (doc_id, score, category, snippet))

# --- Step 8: Inverted index lookup vs linear scan ---
# Linear scan reads every document; the index reads only the posting list of the term.
term = "footbal"
scanned_linear = 0
found_linear = set()
for doc_id, terms in doc_terms.items():
    scanned_linear += 1                  # every document must be opened
    if term in terms:
        found_linear.add(doc_id)
posting = index[term]
print("\nQuery term '%s'" % term)
print("Linear scan : documents scanned =", scanned_linear, end=" ")
print("| matches =", len(found_linear))
print("Index lookup: documents touched =", len(posting), "| matches =", len(posting))
print("Same result :", found_linear == set(posting.keys()))






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 2 : INVERTED INDEX AND RETRIEVAL
================================================================================

CODE EXPLANATION
----------------
1. Imports and downloads: math (log), re (letters-only filter), pandas (load CSV), nltk
   (tokenizer, stop words, Porter stemmer). nltk.download fetches punkt_tab and stopwords once.
2. Load data: Option A (active) reads BBC_News.csv from the datasets repo; Option B (commented)
   reads a local copy. df.head(500) keeps the first 500 articles; the row index 0..499 is the
   document id. N = 500 is the collection size used in the TF-IDF formula.
3. preprocess(text): the same pipeline as Practical 1 in one small function: lower case ->
   word_tokenize -> keep only letter tokens and drop stop words -> Porter stem. The same
   function is used for documents AND for queries, so both sides match.
4. Build the inverted index: index is a dict {term: {doc_id: tf}}. For every document we
   preprocess the text and, for every term, add 1 to index[term][doc_id]. So each term points
   to its "posting list" (documents containing it plus the count in each). doc_terms keeps the
   processed word list of each document, used only by the linear scan in the last step.
5. Save and load (Step 4b): json.dump writes the index to inverted_index.json in the current
   folder and json.load reads it back. THIS FILE IS THE "INVERTED FILE" of the problem
   statement. JSON keys must be text, so doc ids are saved as "22" and converted back with
   int() when loading. The code checks loaded == index (True) and then uses the loaded copy
   for every query below.
6. Statistics: len(index) = vocabulary size; the sum of posting list sizes = total postings; a
   sample posting list for "footbal"; document frequency df = len(index[term]).
7. Boolean retrieval: docs_with(word) returns the SET of document ids of a term. AND is set
   intersection (&), OR is union (|), NOT is difference (-). The queries are hand typed (demo
   input). show() prints doc id, category and the first 100 characters.
8. Ranked retrieval (TF-IDF): for each query term, idf = log(N / df); for each document in its
   posting list add tf * idf to that document's score. Documents are sorted by score and the top
   5 are printed. Query terms are hand typed (demo input).
9. Index vs linear scan: the linear scan opens all 500 documents to find one term; the index
   reads only the posting list (27 documents). Both give the same answer.

EXPECTED OUTPUT (real run, Option A data: BBC_News.csv, first 500 articles)
----------------------------------------------------------------------
   Documents in the collection (N): 500
   {'business': 128, 'sport': 118, 'politics': 97, 'entertainment': 81, 'tech': 76}
   Example: ['market', 'fall', 'investor', 'sell', 'share']
   Inverted file saved to inverted_index.json; loaded copy equals the built index: True
   Vocabulary size (unique terms): 9699
   Total postings (term-document pairs): 68693
   Posting list of 'footbal': [(22, 1), (27, 1), (32, 1), (45, 1), (46, 4), (73, 2), (77, 4),
      (88, 1) ...]
   Document frequency: market 105, footbal 27, mobil 21, phone 30, game 110, govern 116
   BOOLEAN stock AND market        : 21 matches (e.g. doc 12, 15, 36, 49, 52)
   BOOLEAN football OR rugby       : 42 matches (e.g. doc 2, 8, 22)
   BOOLEAN mobile AND phone AND NOT game : 11 matches (e.g. doc 27, 133, 145, 149, 159)
   RANKED "stock market"   : top doc 52 (score 28.11, business), then 193, 393, 186, 357
   RANKED "football match" : top doc 46 (score 18.47), then 77, 90, 289, 73
   RANKED "mobile phone"   : top doc 340 (score 78.14, tech), then 351, 24, 366, 462
   Linear scan : documents scanned = 500 | matches = 27
   Index lookup: documents touched = 27 | matches = 27 | Same result : True
   (No plot is drawn. Everything is printed text.)

OUTPUT EXPLAINED (printed tables, no graph)
-------------------------------------------
No graph is drawn. Four printed blocks:

A. INDEX STATISTICS AND POSTING LIST
   - What it shows: the size of the inverted index and one example posting list.
   - Vocabulary = number of different stems (keys). Total postings = (term, doc) pairs.
   - A posting pair is (document id, term frequency). df = length of the list.
   - Read: (46, 4) means "footbal" is in doc 46 four times.
   - Common words have a big df (govern, game); rarer words a small df (mobil, footbal).
   - Our result: vocabulary 9699, postings 68693; df: footbal 27, market 105, govern 116.
   - If asked: "A posting list = the documents that contain a term, with its count in each."
   - Common misreading: pair numbers are doc id and count, not scores; footbal is a stem.

B. BOOLEAN QUERY RESULTS
   - What it shows: documents matched by each query (up to 5: id, [category], 100 characters).
   - Each query prints "matches: n" and then the documents.
   - AND = both words (intersection), OR = either word (union), NOT = remove (difference).
   - Read: a plain yes/no list with no ranking; documents come in increasing id order.
   - OR is at least as big as each single word list (footbal alone is 27).
   - Our result: stock AND market 21, football OR rugby 42, mobile AND phone AND NOT game 11.
   - If asked: "AND is intersection, OR is union and NOT is difference of posting lists."
   - Common misreading: the first document shown is not the best match; there is no order.

C. TF-IDF RANKED TABLE
   - What it shows: top 5 documents per query, best score first (id, score, [category], text).
   - Score = sum over query words of tf x log(N / df).
   - tf = count of the word in the document; log(N / df) is big for rare words.
   - Read: higher score = more and rarer query words. Compare scores only inside one query.
   - Our result: top docs: 52 (28.11), 46 (18.47), 340 (78.14) for the three queries.
   - If asked: "TF-IDF ranks high a document with the query words often, and rare in the data."
   - Common misreading: ranking follows words, not category (doc 46 is a politics article).
   - Raw tf has no length normalisation, so a long document repeating the words can win.

D. INDEX LOOKUP VERSUS LINEAR SCAN
   - What it shows: the work done to find "footbal" by two methods.
   - Linear scan = open every document. Index lookup = go straight to the posting list.
   - "Same result : True" means both ways found exactly the same documents.
   - Read: fewer documents touched = less work, with no answer lost.
   - Our result: scan 500 docs, 27 matches; index 27 docs, 27 matches; Same result True.
   - If asked: "The index skips documents without the word: 27 lookups instead of 500 scans."
   - Common misreading: this counts documents, not seconds.
   - The index took time to build once, before any query.

DETAILED THEORY NOTES
---------------------
1. WHAT IS AN INVERTED INDEX?
   - Definition: a data structure that maps every TERM to the list of DOCUMENTS that contain it
     (a posting list). It is "inverted" because a normal (forward) index maps document -> its
     words, and this one maps word -> documents.
   - Type: the core data structure of search engines (Information Retrieval). Not a learning
     algorithm.
   - Intuition: the index at the back of a textbook. To find "photosynthesis" you do not read
     the whole book; you look up the word and jump to the listed pages.
   - Maths in words: for a term t, postings(t) = {(doc, tf)} where tf is how many times t occurs
     in doc. Document frequency df(t) = size of the posting list.
   - Building: read every document once, preprocess, and append the document id to each term's
     list. Search then costs only the sum of the lengths of the query terms' posting lists, not
     the size of the whole collection.

2. TYPES AND VARIANTS
   - Record-level index (term -> doc ids only), word-level / positional index (term -> doc id and
     positions; supports phrase queries such as "stock market"), and frequency index (what we
     built: term -> doc id and tf).
   - Retrieval models: Boolean model (exact match, no ranking), Vector Space Model (TF-IDF with
     cosine similarity), Probabilistic models (BM25), Language models, and neural retrievers.
   - Index compression and skip pointers speed up very large indexes; distributed indexes
     (sharding) are used by web search engines.
   - Libraries that do the same professionally: Lucene, Elasticsearch, Whoosh, Solr.

3. KEY CONCEPTS AND TERMS
   - Corpus / collection : all documents (here 500 BBC articles, N = 500).
   - Term / vocabulary : a processed word; vocabulary = all unique terms (9699 here).
   - Posting list : the list of (doc id, tf) pairs for one term.
   - Posting : one entry of a posting list; total postings = 68693 here.
   - tf (term frequency) : how many times a term appears in one document.
   - df (document frequency) : in how many documents a term appears.
   - idf (inverse document frequency) : log(N / df). Rare terms get a high idf, common terms
     a low idf (a term in every document has idf = log(1) = 0).
   - TF-IDF : tf * idf. High when a term is frequent in this document but rare in the
     collection, so it is a good keyword for that document.
   - Boolean query : terms combined with AND, OR, NOT; a document either matches or not.
   - Query processing : the query goes through the SAME preprocessing as the documents.
   - Stemming effect : "football" is stored as "footbal", so the query is stemmed too.
   - Set operations : AND = intersection, OR = union, NOT = difference of posting sets.
   - Linear scan : checking every document one by one (no index).

4. HOW TO JUDGE THE RESULT
   - Efficiency: documents touched. Linear scan 500 vs index lookup 27 for "footbal"; the gap
     grows with the collection size (for millions of documents it is huge).
   - Boolean results: check that the set arithmetic is right: AND gives fewer matches than
     each single term, OR gives more, NOT removes documents.
   - Ranked results: read the top documents; "mobile phone" returns tech articles about phones,
     "stock market" returns business articles on shares.
   - Formal IR measures (not computed here, no relevance judgements): Precision, Recall,
     F-measure, Average Precision, MAP, nDCG. Precision = relevant retrieved / retrieved;
     Recall = relevant retrieved / all relevant.
   - Real observation: ranking is not perfect. For "football match", the top result is a
     politics article about football fans, and "mobile phone" gives a console article because
     of long documents with many repeats (tf grows with document length).

5. DATA SPLIT / RELEVANT SETUP
   - No train/test split: building an index is not learning. We index the first 500 articles
     so the output is small and quick; the same code works for all 2225.
   - Doc id = row index of the DataFrame, so doc 340 means row 340 of the CSV.
   - The index is built ONCE (offline); queries are answered many times (online).
   - Query and documents must use the same preprocessing, otherwise terms will not match.

6. THE DATASET (BBC NEWS)
   - 2225 BBC articles in 5 categories (business, entertainment, politics, sport, tech); we use
     the first 500: business 128, sport 118, politics 97, entertainment 81, tech 76.
   - Columns: category and text. The category is NOT used to search; it is only printed to let
     us judge whether the results make sense.
   - The text is lower case and has little punctuation, which makes tokenizing easy.

7. ADVANTAGES
   - Very fast search: only the posting lists of the query terms are read.
   - Simple to build with plain dictionaries; easy to explain.
   - Boolean queries are exact and predictable; set operations are cheap.
   - TF-IDF ranking is simple, needs no training, and works well as a baseline.
   - Can be extended with positions (phrase search) and updated incrementally.

8. DISADVANTAGES
   - Needs memory/disk for the index and time to build it; updating needs care.
   - Boolean model gives no ranking and is all-or-nothing (too many or zero results).
   - TF-IDF ignores word order, meaning and synonyms ("car" does not match "automobile").
   - Long documents can get high scores simply because they repeat words (no length
     normalisation in our simple score).
   - Stemming may merge unrelated words (market, marketing -> market).
   - Our inverted file is a plain JSON text file, big and slow to load; real engines use
     compressed binary files and disk structures.

9. REAL-WORLD USES
   - Web search engines (Google, Bing), enterprise and desktop search.
   - Library catalogues, legal and patent search, e-commerce product search.
   - Email and log search, plagiarism detection, document deduplication.
   - Elasticsearch/Lucene based applications and many databases (full text search).

10. COMPARISON WITH RELATED METHODS
   - Inverted index vs linear scan : index reads only matching documents (27 vs 500); scan reads
     all. The index costs extra build time and memory.
   - Inverted index vs forward index : forward = document -> terms (good for analysing one
     document); inverted = term -> documents (good for searching).
   - Boolean vs ranked retrieval : Boolean returns an unordered exact-match set; ranked orders
     documents by relevance score and always returns the best few.
   - TF-IDF vs BM25 : BM25 adds term-frequency saturation and document length normalisation and
     usually ranks better; TF-IDF is simpler.
   - TF-IDF vs word embeddings / neural search : embeddings capture meaning and synonyms but
     need large models; TF-IDF matches exact (stemmed) words only.
   - Inverted index vs database B-tree index : a B-tree finds rows by key value; an inverted
     index finds documents by the words inside them.

LIBRARIES AND FUNCTIONS USED
----------------------------
- math.log : natural logarithm, used in idf = log(N / df).
- re.fullmatch(r"[a-z]+", t) : keeps only tokens made of letters.
- pandas (pd) : pd.read_csv (load CSV from URL or local file); df.head(500) (first 500 rows);
  df.category.value_counts() (rows per category); df.text.items() (pairs of row index and text);
  df.category[d] and df.text[d] (value at row index d).
- nltk.download(name, quiet=True) : fetches punkt_tab and stopwords silently.
- nltk.tokenize.word_tokenize : splits text into tokens.
- nltk.corpus.stopwords.words("english") : stop-word list; we convert it to a set.
- nltk.stem.PorterStemmer().stem : reduces a word to its stem.
- json.dump / json.load : write a dict to a text file and read it back (the inverted file).
- dict : index = {term: {doc_id: tf}}; dict.get(key, default) returns default if the key is
  missing (used for counting); index.get(t, {}) avoids a KeyError for unknown terms.
- set operations : & (AND, intersection), | (OR, union), - (NOT, difference).
- sorted(..., key=lambda x: x[1], reverse=True) : orders (doc, score) pairs by score,
  highest first; [:top] keeps the best ones.
- str formatting "%-8s %d %6.2f" : aligns the printed columns.

DETAILED VIVA Q&A
-----------------
Q1. What is mean by inverted index?
A1. A mapping from each term to the list of documents that contain it (with optional
    frequencies or positions). It is the main structure of search engines because a query
    can jump straight to the documents that contain its words.

Q2. Why is it called "inverted"?
A2. A normal index lists the words of each document (document -> words). Here it is turned
    around: word -> documents.

Q3. What is a posting list?
A3. The list of documents in which one term appears. In our code, index["footbal"] is a dict
    {doc_id: tf}, for example {22: 1, 27: 1, 46: 4, ...}.

Q4. What are steps for creation of Inverted Index?
A4. (1) Collect the documents and give each an id. (2) Preprocess each: tokenize, lower case,
    remove stop words, stem. (3) For every term add the document id (and count) to the
    term's posting list. (4) Store/sort the lists. (5) Use it for queries.

Q5. Why do we preprocess the query in the same way as the documents?
A5. The index contains stems like "footbal". If the query word "football" were not stemmed
    too, it would not be found in the index.

Q6. How does Boolean retrieval work?
A6. Each query term gives a set of documents. AND = intersection, OR = union, NOT = difference
    of these sets. A document either matches or it does not; there is no ranking.

Q7. What is the difference between Boolean and ranked retrieval?
A7. Boolean returns an unordered set of exact matches (can be too big or empty). Ranked
    retrieval gives every matching document a score and returns them best first.

Q8. What is term frequency (tf)?
A8. The number of times a term appears in a document. More occurrences suggest the document is
    more about that term.

Q9. What are document frequency (df) and inverse document frequency (idf)?
A9. df is the number of documents that contain the term. idf = log(N / df) is high for rare
    terms and low for common terms, so common words add little to the score.

Q10. What is TF-IDF and why is it used?
A10. TF-IDF = tf * idf. It is large when a word is frequent in a document but rare in the
     collection, so it highlights words that characterise that document. It is used to rank
     documents for a query.

Q11. Why is idf of a word that appears in all documents zero?
A11. If df = N then log(N / N) = log(1) = 0, so the word cannot help to tell documents apart.

Q12. Why is an inverted index faster than a linear scan?
A12. A linear scan opens every document (500). The index goes directly to the posting list of
    the query term (27 documents for "footbal"). The work depends on the number of matching
    documents, not on the collection size.

Q13. What is the vocabulary size here and what does it mean?
A13. 9699 unique stems in the 500 documents. It is the number of keys (rows) in the index.

Q14. What is stemming and why is it used in indexing?
A14. Reducing words to a root form (football -> footbal, markets -> market) so that different
     forms share one posting list, increasing recall.

Q15. What are stop words and why remove them from the index?
A15. Very common words (the, is, of). They appear in almost every document, make posting lists
     huge and carry little meaning, so removing them saves space and time.

Q16. What are the limits of the Boolean model?
A16. Exact match only, no ranking, no partial match, and users must know the logic. It often
     returns too many or too few documents.

Q17. What are the limits of TF-IDF?
A17. It ignores word order and meaning (synonyms), and long documents may score high just
     because they repeat words. BM25 and neural models fix some of this.

Q18. How would you support phrase queries such as "stock market"?
A18. Store positions of each term in each document (positional index) and check that the
     positions are next to each other.

Q19. What are precision and recall in IR?
A19. Precision = relevant documents retrieved / all documents retrieved. Recall = relevant
     documents retrieved / all relevant documents in the collection.

Q20. What is the space cost of an inverted index?
A20. About the number of postings (68693 here), plus the vocabulary. It is usually much
     smaller than the text if stop words are removed and the lists are compressed.

Q21. What is the time complexity of searching a single term?
A21. Dictionary lookup is O(1) on average, then reading the posting list is O(df). For AND of
     two terms, intersection costs about O(df1 + df2) on sorted lists.

Q22. What would you do to update the index when a new document arrives?
A22. Preprocess the new document and add its id to the posting list of each of its terms.
     Real engines add it to a small new index and merge it later.

Q23. Why did the mobile phone query return a game console article at rank 2?
A23. TF-IDF adds up tf * idf, and that article is long and repeats "mobile" and "phone"
     many times, so it gets a high score. Length normalisation (cosine, BM25) would reduce this.

Q24. How is NOT implemented?
A24. As set difference: documents(A) - documents(B). For "mobile AND phone AND NOT game" we
     take docs(mobile) & docs(phone) and remove docs(game).

Q25. What is the difference between an index in a database and an inverted index?
A25. A database index finds rows by a key value; an inverted index finds documents by the words
     they contain and can store tf/positions for ranking.

Q26. Name real systems that use inverted indexes.
A26. Lucene, Elasticsearch, Solr, Whoosh, and the full-text search of PostgreSQL and MySQL.

Q27. What are built in functions used for index creation?
A27. In the manual: the file functions open, read and seek(0) (seek returns the cursor to the
     beginning of the file). In our code: word_tokenize, stopwords.words, PorterStemmer.stem,
     dict and dict.get, set, sorted and math.log (the last one for the idf in ranking).

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Which line builds the posting list entry for one term?
A1. index[term][doc_id] = index[term].get(doc_id, 0) + 1   (after creating index[term] = {} if
    the term is new). It counts tf per document.

Q2. Write the code for df of a term.
A2. df_t = len(index.get(term, {}))   (size of the posting list).

Q3. Change the collection size to 1000 documents. What do you change?
A3. df = df.head(1000). N is computed from len(df), so the idf formula updates automatically.

Q4. How do you search for "business OR sport"?
A4. docs_with("business") | docs_with("sport").

Q5. What happens if the query word is not in the index?
A5. docs_with returns an empty set because index.get(terms[0], {}) gives {}. If the word is a
    stop word or a number (for example "the"), preprocess returns [] and docs_with returns an
    empty set too (the "if terms else set()" part). In the ranked search the term is skipped
    (if term in index).

Q6. How would you remove the effect of document length in the score?
A6. Divide each document's score by its length: score / len(doc_terms[doc_id]), or use cosine
    similarity or BM25.

Q7. Replace idf = log(N / df) by a smoothed idf.
A7. idf = math.log((N + 1) / (df + 1)) + 1 (avoids zero idf and division problems).

Q8. How would you print the top 10 instead of top 5?
A8. search(query, top=10).

Q9. How can you make the Boolean search accept a whole string like "mobile AND phone"?
A9. Split the string on AND/OR/NOT, look up each word with docs_with and combine the sets with
    &, | and - in order. (We typed the queries as code to keep it simple.)

Q10. What does df.text.items() return?
A10. Pairs (row index, text) for each row; row index is used as the document id.

Q11. Why do we use sets for Boolean retrieval?
A11. Sets give fast union, intersection and difference with &, | and -, which is exactly AND,
     OR, NOT.

Q12. How would you show the five most common terms in the index (by df)?
A12. sorted(index, key=lambda t: len(index[t]), reverse=True)[:5].

Q13. Where is the "inverted file" in your code?
A13. inverted_index.json. json.dump(index, f) writes the dict {term: {doc_id: tf}} to disk and
     json.load reads it back. JSON keys must be strings, so doc ids are converted with int()
     after loading. Then loaded == index is True, so queries give the same result.

Q14. Why do you save the index to a file at all?
A14. Building the index means reading and preprocessing every document, which is slow. The
     saved file lets a search program load the ready index and answer queries without
     rebuilding it, like a real search engine.

END OF VIVA NOTES
'''
