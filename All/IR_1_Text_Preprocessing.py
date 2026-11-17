"""
Practical 1 : Text Preprocessing with NLTK (BBC News article)
=============================================================
INSTRUCTIONS
 1. Install:  pip install pandas matplotlib nltk
              (the nltk.download lines in Step 1 fetch punkt_tab, stopwords, wordnet
              and the POS tagger once; this needs internet only the first time)
 2. Dataset : BBC_News.csv (2225 articles; columns category, text). Option A fetches
              it online from the datasets repo; Option B reads a local copy - see the
              two options in Step 2.
 3. Run     : python IR_1_Text_Preprocessing.py
 4. Steps   : load -> pick one article -> sentence segmentation -> tokenization ->
              lowercase -> remove punctuation/numbers -> remove stop words ->
              stemming vs lemmatization -> POS tagging -> counts at each stage ->
              top 10 stems bar chart
"""

# --- Step 1: Import the libraries and download the NLTK data (one time) ---
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

for pkg in ["punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

# --- Step 2: Load the dataset ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/BBC_News.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Download BBC_News.csv from https://github.com/MrNobody4869/datasets
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("BBC_News.csv")
print("Data shape:", df.shape)

# --- Step 3: Pick ONE article as "the document" ---
# First article of the 'tech' category; we show every stage on this single document.
text = df[df.category == "tech"].text.iloc[0]
print("\nORIGINAL TEXT (first 300 characters):\n", text[:300], "...")
print("Characters in the document:", len(text))

# --- Step 4: Sentence segmentation ---
# Split the document into sentences.
sentences = sent_tokenize(text)
print("\nNumber of sentences:", len(sentences))
print("First sentence:", sentences[0])

# --- Step 5: Tokenization ---
# Split the whole text into word and punctuation tokens.
tokens = word_tokenize(text)
print("\nTokens after tokenization:", len(tokens))
print("First 15 tokens:", tokens[:15])

# --- Step 6: Case change (lowercasing) ---
# Make every token lower case so "The" and "the" become the same word.
lower = [t.lower() for t in tokens]
print("\nAfter lowercasing, first 15:", lower[:15])

# --- Step 7: Remove punctuation and numbers ---
# Keep only tokens made of letters (a-z).
alpha = [t for t in lower if re.fullmatch(r"[a-z]+", t)]
print("\nTokens after removing punctuation/numbers:", len(alpha))
print("First 15:", alpha[:15])

# --- Step 8: Remove stop words ---
# Stop words are very common words (the, is, of) that carry little meaning.
stop_words = set(stopwords.words("english"))
print("\nNumber of English stop words in NLTK:", len(stop_words))
no_stop = [t for t in alpha if t not in stop_words]
print("Tokens after stop-word removal:", len(no_stop))
print("First 15:", no_stop[:15])

# --- Step 9: Stemming (Porter) and Lemmatization (WordNet) ---
# Stemming chops word endings by rules; lemmatization returns the dictionary word.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stems = [stemmer.stem(t) for t in no_stop]
lemmas = [lemmatizer.lemmatize(t) for t in no_stop]
# Show a side by side table for the first 10 different words that stemming changes.
print("\n%-15s %-15s %-15s" % ("WORD", "STEM", "LEMMA"))
shown = []
for w in no_stop:
    if stemmer.stem(w) != w and w not in shown:
        shown.append(w)
        print("%-15s %-15s %-15s" % (w, stemmer.stem(w), lemmatizer.lemmatize(w)))
    if len(shown) == 10:
        break
# Demo input: a few hand-typed words to show the difference clearly.
print("\nDEMO INPUT words:")
for w in ["studies", "running", "better", "companies", "wolves"]:
    print("%-10s stem=%-8s lemma(noun)=%-8s lemma(verb)=%-8s" % (
        w, stemmer.stem(w), lemmatizer.lemmatize(w), lemmatizer.lemmatize(w, pos="v")))

# --- Step 10: POS tagging (extra) ---
# Tag each word of the first sentence as noun, verb, adjective, etc.
first_tokens = word_tokenize(sentences[0])
tags = nltk.pos_tag(first_tokens)
print("\nPOS tags of the first sentence (first 10):", tags[:10])

# --- Step 11: Counts at every stage ---
print("\nTOKEN COUNTS")
print("Raw tokens          :", len(tokens))
print("After lowercase     :", len(lower))
print("After punct/numbers :", len(alpha))
print("After stop words    :", len(no_stop))
print("After stemming      :", len(stems), "| unique stems:", len(set(stems)))
print("After lemmatization :", len(lemmas), "| unique lemmas:", len(set(lemmas)))

# --- Step 12: Top 10 most frequent stems with a bar chart ---
top10 = Counter(stems).most_common(10)
print("\nTop 10 stems:", top10)
names = []
freqs = []
for stem, count in top10:
    names.append(stem)
    freqs.append(count)
plt.bar(names, freqs, color="steelblue")
plt.xlabel("Stem")
plt.ylabel("Frequency")
plt.title("Top 10 stems in the tech article")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 1 : TEXT PREPROCESSING WITH NLTK
================================================================================

CODE EXPLANATION
----------------
1. Imports and downloads: re (regular expressions), pandas (load CSV), matplotlib (bar chart),
   nltk (the text toolkit), Counter (count words). nltk.download(..., quiet=True) fetches the
   data files once: punkt_tab (sentence/word splitter), stopwords (list of common words),
   wordnet (dictionary for the lemmatizer), averaged_perceptron_tagger_eng (POS tagger).
   Internet is needed only the first time; later runs find them on disk.
2. Load data: Option A (active) reads BBC_News.csv from the datasets repo URL. Option B
   (commented) reads a local copy. Columns: category and text. No data is made up.
3. Pick one document: df[df.category == "tech"].text.iloc[0] is the first tech article. All
   stages are shown on this single document so the before/after is easy to follow.
4. Sentence segmentation: sent_tokenize(text) returns a list of sentences (31 here).
5. Tokenization: word_tokenize(text) splits text into word and punctuation tokens (773).
6. Lowercasing: t.lower() for every token. (This BBC file is already stored in lower case,
   so the count does not change, but the step is still part of every real pipeline.)
7. Remove punctuation/numbers: re.fullmatch(r"[a-z]+", t) keeps only tokens made purely of
   letters. This drops "." "," "%" and numbers like "2005" (718 left). A word with a hyphen
   such as high-definition is dropped too, a known limit of this simple rule.
8. Stop-word removal: stopwords.words("english") is a list of 198 common words. Tokens that
   are in this set (kept as a set for fast lookup) are removed (392 left).
9. Stemming and lemmatization: PorterStemmer().stem(w) cuts endings by rules (digital ->
   digit, future -> futur). WordNetLemmatizer().lemmatize(w) looks the word up in WordNet and
   returns a real word (wolves -> wolf). A side-by-side table and a demo word list show the
   difference. The demo list is hand typed (demo input).
10. POS tagging: nltk.pos_tag(word_tokenize(sentence)) labels each word with a tag such as
    NN (noun), NNS (plural noun), IN (preposition), DT (determiner).
11. Counts: a summary of how many tokens remain after each stage.
12. Top 10 stems: Counter(stems).most_common(10) and a matplotlib bar chart.

EXPECTED OUTPUT (real run, Option A data: BBC_News.csv)
----------------------------------------------------------------------
   Data shape: (2225, 2)
   Characters in the document: 4333
   Number of sentences: 31
   Tokens after tokenization: 773
   Tokens after removing punctuation/numbers: 718
   Number of English stop words in NLTK: 198
   Tokens after stop-word removal: 392
   Side by side (word / stem / lemma): future / futur / future, theatre / theatr / theatre,
      digital / digit / digital, moving / move / moving, recorders / record / recorder
   Demo words: studies stem=studi lemma=study | running stem=run lemma(verb)=run |
      better stem=better lemma=better | companies stem=compani lemma=company |
      wolves stem=wolv lemma(noun)=wolf
   TOKEN COUNTS: raw 773 -> lower 773 -> alpha 718 -> no stop words 392 ->
      unique stems 237 | unique lemmas 247
   POS of first sentence: ('tv','NN'), ('future','NN'), ('in','IN'), ('the','DT'),
      ('hands','NNS'), ('of','IN') ...
   Top 10 stems: tv 14, peopl 7, us 7, want 7, brand 7, record 6, technolog 5, network 5,
      channel 5, viewer 4
   (One bar chart opens: the top 10 stems. Numbers are for the first tech article.)

OUTPUT GRAPHS EXPLAINED
-----------------------
One output: a bar chart.

FIGURE 1 - bar chart "Top 10 stems in the tech article"
   - What it plots: the 10 most frequent word stems of ONE article (the first tech article).
   - X axis: Stem (labels turned 45 degrees). Y axis: Frequency in this article.
   - All bars are one blue colour, so there is no legend.
   - Read: tall bar = stem used often, so it hints at the topic; tallest is on the left.
   - Stop words are removed first; otherwise "the" and "of" would be the tallest bars.
   - Our result: tv 14, peopl 7, us 7, want 7, brand 7, record 6, then 5s and viewer 4.
   - If asked: "The chart shows the top 10 stems after preprocessing; tv shows the topic."
   - Common misreading: "peopl" is not a typo (Porter cuts endings); counts cover ONE article.

DETAILED THEORY NOTES
---------------------
1. WHAT IS TEXT PREPROCESSING?
   - Definition: cleaning and converting raw text into a simple, uniform form that a program
     can count, compare, index or feed to a model.
   - Type: a data preparation step of Natural Language Processing (NLP) and Information
     Retrieval (IR). It is not a learning algorithm; it is done BEFORE indexing, searching,
     classification or clustering.
   - Intuition: raw text is messy (capital letters, punctuation, endings like -ing/-s, filler
     words). "Running", "runs" and "Run" all mean about the same thing; preprocessing makes
     them look the same so a search for "run" finds all of them.
   - Maths in words: there is no formula. The pipeline is a chain of functions; the output of
     one is the input of the next. The goal is a smaller vocabulary with the same meaning.
   - Typical pipeline: segmentation -> tokenization -> case change -> remove punctuation and
     numbers -> stop-word removal -> stemming or lemmatization (-> POS tagging, spelling
     correction, normalization as needed).

2. TYPES AND VARIANTS
   - Stemmers : Porter (classic, gentle), Lancaster (aggressive), Snowball (improved Porter,
     many languages), RegexpStemmer (your own pattern).
   - Lemmatizers : WordNet lemmatizer (dictionary based), spaCy lemmatizer.
   - Tokenizers : whitespace split, regex tokenizer, word_tokenize (Treebank style), subword
     tokenizers (BPE, WordPiece) used in modern language models.
   - Normalization options : lower casing, expanding contractions (don't -> do not), mapping
     numbers/dates to one token, removing accents, converting emoji to text.
   - Spelling correction : edit distance based tools (Peter Norvig corrector, TextBlob
     correct(), symspell) replace a misspelt word by the closest dictionary word.

3. KEY CONCEPTS AND TERMS
   - Corpus : the whole collection of documents (here the 2225 BBC articles).
   - Document : one item of the corpus (here one tech article).
   - Sentence segmentation : splitting text into sentences. Full stops are not enough
     ("Dr. Smith", "3.5"), so sent_tokenize uses a trained model (punkt).
   - Token / tokenization : a token is a unit such as a word or punctuation mark; tokenization
     is the splitting. "don't" becomes "do" and "n't" in word_tokenize.
   - Case change (case folding) : converting to lower case so that "The" = "the".
   - Stop words : very frequent words with little meaning (the, is, of, and). Removing them
     shrinks the data and removes noise. NLTK's English list has 198 words.
   - Stemming : crude rule-based removal of suffixes to get a stem. The stem may not be a real
     word (studies -> studi, companies -> compani). Fast.
   - Lemmatization : reducing a word to its lemma (dictionary form) using vocabulary and
     grammar, so it is a real word (studies -> study, wolves -> wolf). Slower but cleaner.
   - POS tag : part of speech label of a word (noun NN, verb VB, adjective JJ, adverb RB).
     The lemmatizer needs POS to be exact: lemmatize("running", pos="v") gives "run" but
     the default noun gives "running".
   - Text normalization : bringing text to one standard form (case, spelling, numbers,
     abbreviations, contractions).
   - Vocabulary : the set of unique terms left after preprocessing (237 unique stems here).
   - Regular expression : a pattern to match text; r"[a-z]+" means "one or more letters".

4. HOW TO JUDGE THE RESULT (no accuracy for this practical)
   - Token counts at each stage: 773 -> 718 -> 392 show how much each step removes.
   - Vocabulary size: stemming gave 237 unique terms, lemmatization 247; both are smaller than
     the number of distinct raw words, which is the purpose.
   - Top terms: frequent stems (tv, peopl, brand, technolog) match the article topic, so the
     cleaning kept the meaning.
   - Over-stemming : different words cut to the same stem (university and universe). Under-
     stemming : related words keep different stems.
   - In real projects the true judge is the final task (search quality, classifier accuracy).

5. DATA SPLIT / RELEVANT SETUP
   - No train/test split. Preprocessing is a fixed set of rules, not learned from data, so
     there is nothing to train. (If a step IS learned, e.g. a vocabulary or TF-IDF, it must be
     fitted on training data only.)
   - Setup needed: NLTK data downloads (punkt_tab, stopwords, wordnet, tagger), done once.
   - Order matters: tokenize before removing stop words; lowercase before stop-word lookup
     (the list is lower case); lemmatize before or instead of stem, not usually both.

6. THE DATASET (BBC NEWS)
   - 2225 news articles from the BBC, 5 categories: business, entertainment, politics, sport,
     tech. Columns: category and text.
   - The text in this copy is already lower case and has extra spaces and no punctuation in
     some places (for example "tv future in the hands of viewers ..."), so the sentence
     splitter finds fewer true sentences than a normally written text would give.
   - We use the first article of category tech: 4333 characters, 773 tokens.
   - The same dataset is reused in Practical 2 (inverted index).

7. ADVANTAGES OF PREPROCESSING
   - Smaller vocabulary, so smaller index and faster search.
   - Different forms of a word match each other (better recall in IR).
   - Removes noise, so models focus on meaningful words.
   - Gives consistent input for classification, clustering and indexing.

8. DISADVANTAGES / RISKS
   - Information can be lost: removing stop words breaks phrases such as "to be or not to be";
     removing numbers loses facts; lower casing confuses "US" (country) and "us" (pronoun).
   - Stemming can over-stem or produce non-words; lemmatization is slower and needs POS tags.
   - Language dependent: stop-word lists and stemmers differ for each language.
   - Simple tokenizers fail on hyphens, emails, URLs, hashtags and emoji.

9. REAL-WORLD USES
   - Search engines and library search (index building).
   - Spam filters and sentiment analysis.
   - Chatbots and question answering.
   - Topic modelling, document clustering, plagiarism detection.
   - Machine translation and text summarisation.

10. COMPARISON WITH RELATED METHODS
   - Stemming vs Lemmatization : stemming is fast, rule-based, may give non-words (studi);
     lemmatization is slower, dictionary-based, gives real words (study) and uses POS.
   - Porter vs Lancaster stemmer : Porter is milder; Lancaster is more aggressive and may
     over-stem.
   - Tokenization by split() vs word_tokenize : split() only breaks on spaces so "time." keeps
     its dot; word_tokenize separates punctuation and handles "don't".
   - Stop-word removal vs TF-IDF weighting : removal deletes common words completely; TF-IDF
     keeps them but gives them a low weight automatically.
   - Classic preprocessing vs modern language models : transformers (BERT) use subword tokens
     and need little cleaning; classic IR and bag-of-words models need this full pipeline.

LIBRARIES AND FUNCTIONS USED
----------------------------
- re : regular expressions. re.fullmatch(pattern, string) is true only if the WHOLE string
  matches; r"[a-z]+" = only letters.
- pandas (pd) : pd.read_csv loads the CSV (URL or local file); df.shape gives rows and columns;
  df[df.category == "tech"] filters rows; .text selects the text column; .iloc[0] takes the
  first row.
- matplotlib.pyplot (plt) : bar (bar chart), xlabel/ylabel/title (labels), xticks(rotation=45)
  (tilt labels), tight_layout (fix spacing), show (display).
- nltk : the Natural Language Toolkit. nltk.download(name, quiet=True) fetches data silently.
- nltk.tokenize.sent_tokenize : splits text into sentences (needs punkt_tab).
- nltk.tokenize.word_tokenize : splits text into word and punctuation tokens.
- nltk.corpus.stopwords.words("english") : the list of English stop words.
- nltk.stem.PorterStemmer().stem(word) : Porter stemming.
- nltk.stem.WordNetLemmatizer().lemmatize(word, pos="n") : lemmatization; pos can be "n"
  noun (default), "v" verb, "a" adjective, "r" adverb.
- nltk.pos_tag(tokens) : returns a list of (word, tag) pairs (needs the tagger data).
- collections.Counter : counts items; most_common(10) returns the 10 most frequent as
  (item, count) pairs.
- list comprehension [t for t in tokens if ...] : builds a filtered list in one line.

DETAILED VIVA Q&A
-----------------
Q1. What are the different NLTK libraries?
A1. NLTK has many parts: nltk.tokenize (sent_tokenize, word_tokenize), nltk.corpus (stopwords,
    wordnet, built-in text collections), nltk.stem (PorterStemmer, LancasterStemmer,
    SnowballStemmer, WordNetLemmatizer), nltk.tag (pos_tag), nltk.chunk (chunking), nltk.parse
    (parsers), nltk.classify (Naive Bayes and other classifiers), nltk.probability
    (FreqDist), nltk.metrics (edit distance) and nltk.sentiment (VADER).

Q2. How to remove stop words from the file?
A2. Load the list with stopwords.words("english"), turn it into a set, tokenize the text,
    lower case the tokens, and keep only tokens that are not in the set:
    [t for t in tokens if t not in stop_words]. For a file, read the text first with
    open("file.txt").read() and then apply the same steps.

Q3. What is meant by stemming?
A3. Stemming reduces a word to its stem (root-like form) by chopping suffixes using rules, for
    example running -> run, studies -> studi. The stem need not be a real word. It is fast
    and used in search engines so that different forms match.

Q4. What is meant by lemmatization?
A4. Lemmatization reduces a word to its lemma, the proper dictionary form, using a vocabulary
    and the word's part of speech: studies -> study, wolves -> wolf, better (adjective) ->
    good. The result is always a real word.

Q5. What is text preprocessing and why is it needed?
A5. It is cleaning raw text (split, lower case, remove noise, reduce words to base forms) so
    that the same idea is written in one form. It shrinks the vocabulary and improves search
    and machine learning results.

Q6. What is sentence segmentation?
A6. Splitting a document into sentences. It is harder than splitting at full stops because of
    "Mr.", "e.g." and decimals, so NLTK uses the trained punkt model.

Q7. What is tokenization? How is it different from segmentation?
A7. Tokenization breaks text into tokens (words, punctuation). Segmentation breaks text into
    sentences. Segmentation works at sentence level, tokenization at word level.

Q8. Why do we convert text to lower case?
A8. So "Apple", "APPLE" and "apple" count as one term. It cuts vocabulary size. The cost is
    that names like "US" and "us" become the same.

Q9. What are stop words? Give examples.
A9. The most common words of a language that carry little meaning for search: the, is, a, of,
    and, in, to. NLTK's English list has 198 words.

Q10. Why do we remove punctuation and numbers?
A10. They are usually noise for topic-based tasks and would create many useless tokens. If
     numbers matter (prices, dates) we should keep or normalise them instead.

Q11. Difference between stemming and lemmatization?
A11. Stemming = fast, rule based, may give non-words (compani). Lemmatization = slower,
     dictionary + POS based, always gives real words (company). Use stemming for speed in
     search, lemmatization when meaning and readable words matter.

Q12. Which stemmer did you use and what are the others?
A12. PorterStemmer. Others: LancasterStemmer (more aggressive), SnowballStemmer (improved
     Porter, supports many languages) and RegexpStemmer.

Q13. What is POS tagging?
A13. Assigning a grammatical label (noun NN, verb VB, adjective JJ ...) to every word using
     nltk.pos_tag. It helps lemmatization and later steps like named entity detection.

Q14. Why does the lemmatizer need a POS?
A14. The same word has different lemmas by role: "running" as a verb -> run, as a noun ->
     running; "better" as an adjective -> good. Default pos is noun, so verbs are not reduced
     unless pos="v" is given (see our demo words).

Q15. What is text normalization?
A15. Converting text to a standard form: lower casing, fixing spelling, expanding
     contractions, unifying numbers and dates, removing accents. Stemming and lemmatization are
     also forms of normalization.

Q16. What is spelling correction and how is it done?
A16. Replacing misspelt words with the most likely correct word. Usual method: find dictionary
     words with the smallest edit distance (insert, delete, replace, swap letters) and pick
     the most frequent one. Tools: TextBlob, Norvig's corrector, symspell.

Q17. What does the tokens count 773 -> 718 -> 392 tell us?
A17. 773 raw tokens include punctuation and numbers; removing them leaves 718 words; removing
     stop words leaves 392 meaningful words, about half of the original. So stop words and
     punctuation are about half of the text.

Q18. Why is the number of unique stems (237) smaller than unique lemmas (247)?
A18. Stemming cuts more aggressively and merges more forms into one stem (digital and digit,
     recorders and record), so fewer unique terms remain. Lemmatization only merges true
     inflections (plural to singular).

Q19. Is the order of the steps important?
A19. Yes. Tokenize first; lower case before the stop-word check (the list is lower case); remove
     stop words before stemming so that you do not stem words you will throw away.

Q20. What are the problems of stop-word removal?
A20. It can destroy meaning in phrases ("to be or not to be", "The Who"), negations ("not
     good" loses "not") and hurts tasks like sentiment analysis. Always check the task.

Q21. What is a corpus and what is vocabulary?
A21. Corpus = the whole collection of documents (2225 BBC articles). Vocabulary = the set of
     unique terms after preprocessing.

Q22. What is over-stemming and under-stemming?
A22. Over-stemming: different words reduced to the same stem (university, universe ->
     univers). Under-stemming: related words keep different stems (data and datum).

Q23. What is a regular expression? Which one did you use?
A23. A pattern language for matching text. We used r"[a-z]+" with fullmatch, which keeps only
     tokens consisting entirely of letters.

Q24. Why is "us" in our top 10 stems?
A24. In this data every "us" is the country "US" written in lower case ("the us leads", "the
     us s tivo"), not the pronoun. Lower casing turned it into "us", a live example of the
     case-folding risk in theory section 8. NLTK's stop-word list does not contain "us"
     either way, so it survives. We can add our own stop words with a set union.

Q25. How does preprocessing help an information retrieval system?
A25. Query and documents are processed the same way, so "running" in the query matches "runs"
     in a document. The index becomes smaller and faster, and recall improves.

Q26. Name some tools other than NLTK.
A26. spaCy, TextBlob, Gensim, Stanford CoreNLP, scikit-learn text utilities, Hugging Face
     tokenizers.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Write the line that removes stop words.
A1. no_stop = [t for t in alpha if t not in stop_words]   (stop_words is a set built from
    stopwords.words("english")).

Q2. What happens if you do not lower case before removing stop words?
A2. Capitalised stop words such as "The" would not match the lower-case list and would stay.
    (This dataset is already lower case, so here there is no visible change.)

Q3. Change PorterStemmer to LancasterStemmer. What happens?
A3. Import LancasterStemmer from nltk.stem and use it instead. Stems become shorter and more
    aggressive (for example "maximum" -> "maxim"); the number of unique stems usually drops.

Q4. How do you lemmatize verbs properly?
A4. Pass the POS: lemmatizer.lemmatize("running", pos="v") gives "run". Without pos the
    word is treated as a noun and stays "running".

Q5. Change the code to use the first 'sport' article.
A5. Replace "tech" with "sport": text = df[df.category == "sport"].text.iloc[0].

Q6. How do you keep numbers in the text?
A6. Change the regex to r"[a-z0-9]+" (letters and digits) in Step 7.

Q7. How do you add your own stop words?
A7. stop_words = set(stopwords.words("english")) | {"said", "also"} (union with a set).

Q8. What is the output of sent_tokenize on a text with 3 full stops?
A8. Usually 3 sentences, but not always: abbreviations like "Dr." can create wrong splits, and
    text without capital letters (like this dataset) can reduce the number found.

Q9. How do you get the top 5 instead of the top 10 stems?
A9. Counter(stems).most_common(5) and use top5 in the bar chart.

Q10. Why is df[df.category == "tech"] used?
A10. It is a boolean filter: keeps only rows whose category is tech. .text.iloc[0] then takes
     the text of the first such row.

Q11. How can you remove stop words from a text file?
A11. words = open("file.txt").read().split(); then filter with the stop word set. For better
     tokens use word_tokenize on the file text.

Q12. What does quiet=True do in nltk.download?
A12. It suppresses the progress messages. The data is still downloaded (once) if missing.

END OF VIVA NOTES
'''
