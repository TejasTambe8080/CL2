"""
Practical IR-5 : Agglomerative Hierarchical Clustering on Credit Card Customers
===============================================================================
INSTRUCTIONS
 1. Install:  pip install numpy pandas matplotlib scikit-learn scipy
 2. Dataset : CC_GENERAL.csv (Kaggle "Credit Card Dataset for Clustering": 8950
              cardholders, a CUST_ID column and 17 numeric usage columns).
              Option A reads it online from the datasets repo; Option B reads a
              local copy - see the two options in Step 2.
 3. Run     : python IR_5_Agglomerative_Clustering.py
 4. Steps   : load -> drop CUST_ID -> fill nulls (median) -> StandardScaler +
              normalize -> PCA (2) -> 6-point worked example (merge table)
              -> dendrogram (ward) -> fit k = 2..6 and plot -> silhouette table
              + bar chart -> best k -> cluster profile -> compare
              single / complete / average / ward linkage
"""

# --- Step 1: Import the libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# --- Step 2: Load the dataset ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/CC_GENERAL.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Download CC_GENERAL.csv from https://github.com/MrNobody4869/datasets
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("CC_GENERAL.csv")
print("Data shape:", df.shape)
print(df.head())

# --- Step 3: Drop the identifier and count the missing values ---
# CUST_ID is only a name tag for the customer, not a feature, so do not cluster it.
df = df.drop(columns="CUST_ID")
print("\nCUST_ID dropped (an identifier, not a feature). Columns left:", df.shape[1])
print("Missing values per column (only non-zero shown):")
nulls = df.isnull().sum()
print(nulls[nulls > 0])

# --- Step 4: Fill the missing values with the column median ---
# The manual used fillna(method='ffill') which is removed in new pandas and copies
# another customer's value; the median of the column is safer and works on current
# pandas.
df = df.fillna(df.median())
print("Missing values after filling:", int(df.isnull().sum().sum()))

# --- Step 5: Scale and normalize ---
# StandardScaler: the columns have very different units (0-1 up to 76,000), so each
# column is turned into mean 0 and std 1; otherwise big-valued columns would decide
# every distance.
X_scaled = StandardScaler().fit_transform(df)
# normalize: then each customer (row) is scaled to unit length, so only the spending
# pattern counts, not how large the numbers are overall (this is the second step in
# the manual).
X_normalized = normalize(X_scaled)

# --- Step 6: PCA down to 2 components ---
# 17 columns cannot be drawn, so PCA keeps the 2 directions with the most variance.
# It also makes the dendrogram cheap. The cost: only part of the information is kept.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
variance_ratio = pca.explained_variance_ratio_
print("\nPCA explained variance ratio:", variance_ratio.round(3))
print("Total variance kept by 2 components:", round(variance_ratio.sum(), 3))
print("(So about half of the information is lost; "
      "the clusters below are clusters of this 2-D view.)")

# --- Step 7: Tiny worked example - the algorithm by hand on 6 real customers ---
# Start with 6 clusters (one per customer), then repeatedly merge the two closest.
small = X_pca[:6]
print("\nFirst 6 customers (P1, P2):\n", small.round(3))
merge_table = linkage(small, method="ward")
print("Merge table: [cluster A, cluster B, distance, size of new cluster]")
print(merge_table.round(3))
# Row i of the table creates a new cluster with id 6 + i (ids 0-5 are single customers).

# --- Step 8: Dendrogram for all customers (ward linkage) ---
# Ward on 8950 points is fine (about 8950 x 8950 / 2 distances, roughly 320 MB).
Z = linkage(X_pca, method="ward")
# Show only the last 30 merges so the picture stays readable.
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30)
plt.title("Dendrogram (ward, last 30 merges)")
plt.xlabel("Cluster (number of customers in brackets)")
plt.ylabel("Merge distance")
plt.show()

# How to read k from the dendrogram, by eye: draw a horizontal line through the
# longest vertical lines (the ones no merge crosses) and count the vertical lines it
# cuts. The tallest line is the last merge, so a cut just below it leaves 2 clusters.
# The silhouette scores in Step 10 confirm k.

# --- Step 9: Fit Agglomerative Clustering for k = 2 to 6 and plot all in one figure ---
# Default AgglomerativeClustering uses ward linkage and euclidean distance (manual).
all_labels = {}
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for i, k in enumerate(range(2, 7)):
    all_labels[k] = AgglomerativeClustering(n_clusters=k).fit_predict(X_pca)
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=all_labels[k], cmap="rainbow", s=5)
    axes[i].set_title(f"k = {k}")
    axes[i].set_xlabel("P1")
    axes[i].set_ylabel("P2")
axes[5].axis("off")
plt.tight_layout()
plt.show()

# --- Step 10: Silhouette score for each k, then pick the best k ---
# Silhouette: higher = better separated clusters (range -1 to +1).
ks = list(range(2, 7))
scores = [silhouette_score(X_pca, all_labels[k]) for k in ks]
print("\nSilhouette score for each k (ward):")
for k, s in zip(ks, scores):
    print(f"  k = {k}   {s:.3f}")
plt.bar(ks, scores)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title("Silhouette per k")
plt.show()
best_k = ks[int(np.argmax(scores))]
print("Best k by silhouette:", best_k, "(score", round(max(scores), 3), ")")

# --- Step 11: Profile the clusters of the best k on the ORIGINAL (unscaled) columns ---
df["Cluster"] = all_labels[best_k]
cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE",
        "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT"]
profile = df.groupby("Cluster")[cols].mean().round(2)
profile.insert(0, "Size", df["Cluster"].value_counts().sort_index())
print(f"\nCluster profile for k = {best_k} (size and mean of original columns):")
print(profile.to_string())

# Describe each cluster in words by comparing its means with the average of all.
overall = df[cols].mean()
print("\nWhat each cluster looks like (compared with the average customer):")
for c, row in profile.iterrows():
    if row["PURCHASES"] > overall["PURCHASES"]:
        buys = "high purchases"
    else:
        buys = "low purchases"
    if row["CASH_ADVANCE"] > overall["CASH_ADVANCE"]:
        cash = "uses cash advances a lot"
    else:
        cash = "few cash advances"
    if row["PRC_FULL_PAYMENT"] > overall["PRC_FULL_PAYMENT"]:
        pay = "pays in full more than average"
    else:
        pay = "pays in full less than average"
    print(f"  Cluster {c} ({int(row['Size'])} customers): {buys}, {cash}, {pay}")

# --- Step 12: Compare the four linkage methods (k = best k, on the 2-D PCA data) ---
print(f"\nLinkage comparison (k = {best_k}):")
for method in ["single", "complete", "average", "ward"]:
    lab = AgglomerativeClustering(n_clusters=best_k, linkage=method).fit_predict(X_pca)
    print(f"  {method:9s} silhouette = {silhouette_score(X_pca, lab):.3f}")






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL IR-5 : AGGLOMERATIVE HIERARCHICAL CLUSTERING
================================================================================

CODE EXPLANATION
----------------
1. Imports: pandas and numpy for data, matplotlib for plots, StandardScaler and normalize for
   preprocessing, PCA to reduce 17 columns to 2, AgglomerativeClustering (sklearn) for the
   model, silhouette_score for quality, scipy's linkage and dendrogram for the merge table and
   the tree picture.
2. Load data: Option A (active) reads CC_GENERAL.csv from the datasets repo; Option B
   (commented) reads a local copy. 8950 credit card customers, 18 columns.
3. Drop CUST_ID and count nulls: CUST_ID is an identifier (a name tag), not a feature, so it is
   dropped. Two columns have missing values: MINIMUM_PAYMENTS (313) and CREDIT_LIMIT (1).
4. Fill nulls: the manual used fillna(method='ffill'), which copies the value of the customer
   in the row above (a different person) and no longer works on current pandas. We fill each
   column with its MEDIAN instead: df.fillna(df.median()). Median is not pulled by outliers.
5. Scale and normalize (the manual's two steps): StandardScaler gives every column mean 0 and
   std 1, because the columns have wildly different units (0-1 up to 76,000). normalize()
   then makes each customer (row) a vector of length 1, so the pattern of use matters, not
   the overall size of the numbers.
6. PCA with 2 components: keeps the two directions of largest variance so the data can be
   plotted and the dendrogram is cheap. The explained variance ratio is printed: 0.290 and
   0.184, total 0.474. So only about 47% of the information is kept (honest limitation).
7. Worked example: the first 6 customers (their P1, P2 values) go through scipy
   linkage(method="ward"). The merge table shows which two clusters are joined at each step,
   their distance and the size of the new cluster: the whole algorithm in miniature.
8. Dendrogram: linkage() on all 8950 PCA points, dendrogram() draws the tree with
   truncate_mode="lastp", p=30 (last 30 merges only). Nothing is
   computed to pick k: k is read from the dendrogram by eye (a horizontal line through the
   longest vertical lines) and then confirmed by the silhouette scores of Step 10.
9. Fit k = 2..6: AgglomerativeClustering(n_clusters=k) (default ward + euclidean, as in the
   manual) for each k; the five scatter plots (coloured by cluster) are drawn in ONE figure
   with subplots.
10. Silhouette: a score for each k is printed as a table and drawn as a bar chart; the best k
    is taken with max / argmax (not typed by hand).
11. Profile: the best k's clusters are described using the ORIGINAL unscaled columns: size and
    mean BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS, PRC_FULL_PAYMENT. A small
    loop compares each cluster mean with the overall mean and prints a plain description.
12. Linkage comparison: single, complete, average and ward are fitted at the best k on the 2-D
    PCA data and compared by silhouette.

EXPECTED OUTPUT (real run, Option A data)
----------------------------------------------------------------------
   Data shape: (8950, 18)
   CUST_ID dropped; nulls: CREDIT_LIMIT 1, MINIMUM_PAYMENTS 313; after median fill: 0
   PCA explained variance ratio: 0.290 0.184 ; total kept 0.474
   6-customer worked example, ward merge table [A, B, distance, size]:
      [0 4 0.211 2] [2 5 0.500 2] [3 6 0.551 3] [1 7 1.156 3] [8 9 1.443 6]
   Dendrogram (picture only): the tallest vertical line is the last merge (height about 62),
      so a cut just below it gives k = 2; silhouette (next line) confirms it.
   Silhouette for each k (ward): 2 0.421 | 3 0.400 | 4 0.385 | 5 0.356 | 6 0.345
   Best k by silhouette: 2 (0.421)
   Cluster profile for k = 2 (size, BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS,
   PRC_FULL_PAYMENT):
      0   4045   1924.70    184.82   1614.17   4028.48   1450.35   0.04
      1   4905   1267.41   1678.10    454.96   4878.42   1966.35   0.25
   Cluster 0: low purchases, uses cash advances a lot, pays in full less than average
              (cash-advance users: they borrow cash on the card and rarely clear the bill)
   Cluster 1: high purchases, few cash advances, pays in full more than average
              (card users for shopping: they buy a lot and pay it back)
   Linkage comparison (k=2): single 0.194 | complete 0.308 | average 0.427 | ward 0.421
   (Three plots open: the dendrogram, the 5 scatter plots in one figure
   and the silhouette bar chart. Cluster numbers can differ on another scipy / sklearn
   version; the groups stay the same.)

OUTPUT GRAPHS EXPLAINED
-----------------------
Three figures: dendrogram (Fig 1), five scatter plots (Fig 2), silhouette bars (Fig 3).

GRAPH 1 : DENDROGRAM (Figure 1, title "Dendrogram (ward, last 30 merges)")
- Plots: the merge tree of ward clustering, cut to the last 30 merges so it stays readable.
- x = clusters; a number in brackets is how many customers are inside, for example (679).
- y = "Merge distance": how far apart two clusters were when they joined.
- Each U shape is one merge. Its top bar is the distance; low = joined early, high = late.
- Pick k by cutting across the longest vertical lines that no merge crosses; count lines cut.
- OUR result: the last merge is at about 62, so a cut just below it gives k = 2.
- If asked: "The longest vertical line is the last merge; cutting under it gives 2 clusters."
- Misreading: the left-to-right order of leaves means nothing; only the join height counts.

GRAPH 2 : SCATTER PLOTS FOR k = 2 TO 6 (Figure 2, grid of 5 plots, the 6th box is empty)
- Plots: the same 8950 customers drawn 5 times, coloured by the cluster for each k.
- x = P1 and y = P2, the two PCA components. One dot is one customer.
- Colours are only labels for groups; they do not rank or score anything.
- Read the set together: from k = 2 to 6 each plot splits one group of the plot before it.
- It is the same tree cut at lower and lower heights, so the groups are nested.
- OUR result: one round cloud with no gap, so groups touch (silhouette 0.421 to 0.345).
- If asked: "Same data cut at different levels; the groups touch, so a small k is preferred."
- Misreading: a colour in one plot is not the same cluster in another; PCA keeps only 47.4 %.

GRAPH 3 : SILHOUETTE BAR CHART (Figure 3, title "Silhouette per k")
- Plots: one bar per k from 2 to 6; the height is the silhouette score for that k.
- x = "Number of clusters", y = "Silhouette score", from -1 to +1.
- Higher means tight clusters that are well apart; near 0 means they overlap.
- A negative bar means many points sit in the wrong cluster.
- Read it by picking the tallest bar as the best k.
- OUR result: the bars fall from 0.421 at k = 2 to 0.345 at k = 6, so the best k is 2.
- If asked: "Silhouette is highest at k = 2, so we choose 2; it agrees with the dendrogram."
- Misreading: the tallest bar is not perfect (0.421 is only fair), and bars show no members.

DETAILED THEORY NOTES
---------------------
1. WHAT IS AGGLOMERATIVE HIERARCHICAL CLUSTERING?
   - Definition: a BOTTOM-UP clustering method. Every point starts as its own cluster; the two
     closest clusters are merged again and again until only one cluster is left. The history
     of merges is a tree called a dendrogram.
   - Type: UNSUPERVISED learning (no labels).
   - Intuition: like joining nearby villages into towns, towns into districts, districts into a
     state. Cutting the tree at a chosen height gives the clusters at that level.
   - Algorithm steps:
        (1) start with n clusters, one per point;
        (2) compute the distance between every pair of clusters (distance matrix);
        (3) merge the two closest clusters;
        (4) update the distances to the new cluster (this is where linkage matters);
        (5) repeat 3-4 until one cluster remains;
        (6) draw the dendrogram and cut it to get k clusters.
   - Maths in words: Euclidean distance between points; for clusters the "linkage" rule
     defines distance. Ward linkage merges the pair that gives the smallest increase in total
     within-cluster variance (sum of squared distances to the cluster mean).

2. TYPES AND VARIANTS
   - Agglomerative (bottom-up, this practical) vs Divisive (top-down: start with one cluster
     and split).
   - Linkage types:
        single   : distance between the two CLOSEST points of two clusters (can chain).
        complete : distance between the two FARTHEST points (compact clusters).
        average  : average of all pair distances.
        ward     : minimise increase in variance (compact, similar size clusters; default
                   choice for Euclidean data).
   - Other clustering: K-Means (partition), DBSCAN (density), Gaussian Mixture (soft).

3. KEY CONCEPTS AND TERMS
   - Dendrogram : tree diagram; height of a join = distance at which clusters merged.
   - Linkage : rule to measure distance between two clusters.
   - Linkage matrix (scipy) : each row = one merge: [cluster A, cluster B, distance, new size].
     New clusters get ids n, n+1, ...
   - Cutting the dendrogram : a horizontal line; the number of vertical lines it crosses = k.
   - Reading k from the dendrogram : draw a horizontal line through the longest vertical lines
     (no merge crosses them); natural clusters are far apart. Here the tallest line is the last
     merge (height about 62), so the cut leaves k = 2. The picture is read by eye; the silhouette
     scores then confirm the choice.
   - Truncation (lastp) : show only the last p merges for readability.
   - StandardScaler : gives every feature mean 0 and standard deviation 1.
   - normalize : divides every row by its length, so each customer is a unit-length vector.
   - PCA : projects the data on the directions of largest variance (principal components).
   - Explained variance ratio : share of the total variance kept by each component.
   - Missing value imputation : filling nulls; here with the column median.
   - Silhouette score : internal quality measure (see below).
   - Cluster profile : the mean of the original columns inside each cluster.

4. EVALUATION METRICS
   - Silhouette score = average of (b - a) / max(a, b), where a = mean distance of a point to its
     own cluster and b = mean distance to the nearest other cluster. Range -1 to +1; near 1
     means well separated, near 0 overlapping, negative = probably wrongly placed. Here the
     best value is 0.421 (k = 2): only a moderate separation, real customers form a
     continuous cloud, not clearly separate islands.
   - Dendrogram : visual check for k (longest vertical lines).
   - Cluster profile : check that clusters make business sense (means of the original columns).
   - No accuracy / confusion matrix: there are no true labels in clustering.
   - The silhouette here is computed on the 2-D PCA points, so it judges the clusters of the
     2-D view, not of the full 17-D data.

5. DATA SPLIT / RELEVANT SETUP
   - No train / test split: clustering is unsupervised, all 8950 customers are clustered.
   - StandardScaler and PCA are fitted on all data (nothing to leak because there is no target).
   - Ward needs Euclidean distance, so scaling matters. The method stores or computes all
     pairwise distances: memory and time grow about with n squared. For 8950 points that is
     about 40 million distances (roughly 320 MB as 8-byte numbers), still fine; for 100,000
     points it would be 5 billion and would not fit in normal memory.
   - Preprocessing order (as in the manual): drop id -> fill nulls -> scale -> normalize -> PCA.

6. THE DATASET
   - CC_GENERAL.csv (Kaggle "Credit Card Dataset for Clustering", the dataset named in the
     lab manual): the credit card usage of about 8950 active cardholders over the last 6
     months. 8950 rows, 18 columns: CUST_ID plus 17 numeric behaviour columns. No target /
     label column, so the task is unsupervised clustering (customer segmentation).
   - The 17 columns:
        BALANCE                          : money left in the account available for purchases
        BALANCE_FREQUENCY                : how often the balance is updated (0 to 1, 1 = often)
        PURCHASES                        : total amount spent on purchases
        ONEOFF_PURCHASES                 : largest single purchase amount in one go
        INSTALLMENTS_PURCHASES           : amount spent on purchases paid in instalments
        CASH_ADVANCE                     : cash taken in advance from the card
        PURCHASES_FREQUENCY              : how often purchases are made (0 to 1)
        ONEOFF_PURCHASES_FREQUENCY       : how often one-go purchases are made (0 to 1)
        PURCHASES_INSTALLMENTS_FREQUENCY : how often instalment purchases are made (0 to 1)
        CASH_ADVANCE_FREQUENCY           : how often cash advances are taken
        CASH_ADVANCE_TRX                 : number of cash advance transactions
        PURCHASES_TRX                    : number of purchase transactions
        CREDIT_LIMIT                     : credit limit of the card
        PAYMENTS                         : total payments made by the customer
        MINIMUM_PAYMENTS                 : minimum payments due
        PRC_FULL_PAYMENT                 : share of months in which the bill was paid in full
        TENURE                           : how many months the customer has had the card (6-12)
   - Missing values: MINIMUM_PAYMENTS 313 nulls, CREDIT_LIMIT 1 null; no duplicate rows.
   - Scales differ a lot: frequency columns are 0 to 1 while PURCHASES goes up to about
     49,000 and MINIMUM_PAYMENTS up to about 76,000. That is why scaling is a must.
   - Skew: the money columns are strongly right skewed (mean much larger than median: mean
     PURCHASES 1003 vs median 361; median CASH_ADVANCE is 0 because most customers take none),
     with a few very big spenders. This is one reason the median is used for filling.
   - The manual drops CUST_ID, fills nulls, scales, normalizes, reduces to 2 PCA components
     and compares k = 2 to 6; the code follows the same order. Differences: median fill
     instead of ffill, all plots of k in one figure, and a profile of the clusters.

7. ADVANTAGES
   - No need to choose k in advance: the dendrogram shows all levels.
   - Easy to understand and to draw; good for small data and for finding hierarchy (taxonomy).
   - Deterministic: no random start, the same result every run.
   - Works with any distance and several linkage rules.

8. DISADVANTAGES
   - Slow and memory heavy for big data (O(n^2) memory, about O(n^2 log n) to O(n^3) time).
   - A merge can never be undone (greedy decision).
   - Sensitive to noise and outliers (single linkage chains), and to scaling.
   - Different linkages give different results; ward assumes compact round clusters.
   - Dendrogram becomes unreadable for thousands of points (needs truncation).
   - Here the 2-D PCA view keeps only 47% of the variance, so the clusters are an
     approximation of the real 17-D behaviour.

9. REAL-WORLD USES
   - Customer segmentation (this practical, credit card customers), market research.
   - Document / news clustering and topic hierarchies in information retrieval.
   - Gene expression and biology (phylogenetic trees), social network communities.
   - Image segmentation and organising search results.

10. COMPARISON WITH RELATED METHODS
   - Agglomerative vs K-Means : K-Means needs k first, is fast and random-start dependent;
     agglomerative needs no k up front, is deterministic and gives a hierarchy but is slower.
   - Agglomerative vs DBSCAN : DBSCAN finds arbitrary shapes and noise points; agglomerative
     (ward) forms compact clusters and puts every point in a cluster.
   - Agglomerative vs Divisive : divisive starts from the whole data and splits (top-down); it
     is more costly and less common.
   - Ward vs single : ward gives balanced compact clusters; single links by the nearest
     neighbours and can form long chains or one huge cluster with a few outliers (single
     scored only 0.194 here).
   - Ward vs average : in our run average (0.427) is a hair above ward (0.421); the two are
     close, ward is kept because it is the manual's choice and gives balanced sizes.

LIBRARIES AND FUNCTIONS USED
----------------------------
- pandas (pd) : read_csv loads the file; drop(columns=...) removes CUST_ID; isnull().sum()
  counts nulls; fillna(df.median()) fills nulls; groupby("Cluster")[cols].mean() cluster
  averages; value_counts; insert adds the Size column; head; iterrows loops over rows.
- numpy (np) : argmax finds the index of the highest silhouette score.
- matplotlib.pyplot (plt) : figure, subplots (a grid of plots in one figure), scatter (c =
  cluster number, cmap colours, s = point size), bar, xlabel / ylabel / title, tight_layout,
  show.
- sklearn.preprocessing.StandardScaler : fit_transform scales each column to mean 0, std 1.
- sklearn.preprocessing.normalize : makes every row have length 1 (L2 norm).
- sklearn.decomposition.PCA : n_components=2; fit_transform returns the 2-D points;
  explained_variance_ratio_ gives the variance share of each component.
- sklearn.cluster.AgglomerativeClustering : n_clusters=k, linkage="ward" (default; also
  single, complete, average); fit_predict returns the cluster numbers.
- sklearn.metrics.silhouette_score : internal cluster quality.
- scipy.cluster.hierarchy.linkage : builds the full merge table (method="ward"); returns an
  array with 4 columns.
- scipy.cluster.hierarchy.dendrogram : draws the tree; truncate_mode="lastp" with p=30 shows
  the last 30 merges.
- Python parts : f-string formats (:9s, :.3f), a for loop with zip and enumerate, a dictionary
  all_labels to keep the labels of every k, list comprehension for the scores.

DETAILED VIVA Q&A
-----------------
Q1. What is mean by Agglomerative hierarchical clustering algorithm?
A1. It is a bottom-up clustering algorithm. Each data point starts as its own cluster; the two
    closest clusters are merged repeatedly until one cluster remains. The sequence of merges is
    shown as a dendrogram, which can be cut at any level to get the required number of clusters.

Q2. Which is the readymade function available to build Agglomerative hierarchical clustering
    algorithm?
A2. sklearn.cluster.AgglomerativeClustering for fitting clusters, and
    scipy.cluster.hierarchy.linkage with dendrogram for the merge table and the tree picture.

Q3. Tell me the steps to implement Agglomerative hierarchical clustering algorithm.
A3. Load the data and drop the id column; fill missing values; scale and normalize; reduce with
    PCA; compute the linkage (distance matrix and merges); draw the dendrogram; choose the
    number of clusters by eye from the dendrogram (longest vertical lines); fit AgglomerativeClustering for the
    chosen k values; plot and profile the clusters; evaluate with the silhouette score.

Q4. Applications of Agglomerative hierarchical clustering algorithm.
A4. Customer segmentation, document and news grouping, gene expression analysis, social network
    community detection, image segmentation and building taxonomies.

Q5. What is a dendrogram?
A5. A tree diagram that records the merges. The leaves are the points, the height of every join
    is the distance at which the two clusters merged. Cutting it with a horizontal line gives
    clusters.

Q6. How do you choose the number of clusters from the dendrogram?
A6. Find the longest vertical line that is not crossed by any horizontal merge line, draw a
    horizontal cut through it and count the vertical lines it crosses. Here the
    tallest line is the last merge (height about 62), so the cut leaves k = 2. We read this by
    eye and then confirm it with the silhouette scores.

Q7. What is linkage? Name the types.
A7. The rule that defines the distance between two clusters: single (closest points), complete
    (farthest points), average (mean of all pairs) and ward (smallest increase in variance).

Q8. Why ward linkage here?
A8. Ward tends to make compact clusters of similar size with Euclidean distance, which is what
    we expect for customer groups, and it is the default in the manual. In the comparison table
    ward scored 0.421 and average 0.427 (almost equal), complete 0.308 and single 0.194.

Q9. Agglomerative vs divisive?
A9. Agglomerative starts with n clusters and merges (bottom-up); divisive starts with one
    cluster and splits (top-down). Agglomerative is far more common.

Q10. Why scale the data?
A10. Distances are used to decide merges, so a feature with bigger numbers would dominate. In
     this dataset PURCHASES goes up to about 49,000 while the frequency columns lie between 0
     and 1, so without scaling the money columns would decide everything.
     StandardScaler puts all features on the same scale.

Q11. Do we need a train / test split?
A11. No. Clustering is unsupervised, there is nothing to predict; we cluster all 8950
     customers.

Q12. What does the merge (linkage) table contain?
A12. Four columns per merge: the ids of the two clusters joined, the distance between them, and
     the number of points in the new cluster. New cluster ids start from n (6 in the small
     example, 8950 in the full data).

Q13. What is the silhouette score and what did you get?
A13. It compares how close a point is to its own cluster versus the nearest other cluster, from
     -1 to +1. We got 0.421 for k = 2, 0.400 for k = 3, 0.385 for k = 4, 0.356 for k = 5 and
     0.345 for k = 6: moderate separation, best at k = 2.

Q14. Why does single linkage give poor clusters?
A14. It joins clusters through their nearest pair of points, so a chain of close points links
     everything (chaining effect). It often produces one big cluster and tiny ones, so its
     silhouette is low (0.194 here).

Q15. Why is a merge irreversible?
A15. The algorithm is greedy: once two clusters are merged they are never split again, even if
     a later step shows it was a bad choice.

Q16. Time and memory complexity?
A16. It needs the pairwise distance matrix, so memory is O(n^2) and time between O(n^2 log n)
     and O(n^3). That is why it suits thousands, not millions, of points.

Q17. K-Means vs Agglomerative?
A17. K-Means: k fixed first, fast, random start, round clusters. Agglomerative: no k needed up
     front, deterministic, gives the whole hierarchy, but slower on big data.

Q18. How would you describe the two customer segments found?
A18. Cluster 0 (4045 customers) buys little (mean purchases 184.82) but takes a lot of cash
     advances (1614.17) and rarely pays the bill in full (0.04): cash-advance borrowers.
     Cluster 1 (4905 customers) buys a lot (1678.10), takes few cash advances (454.96) and pays
     in full more often (0.25): shopping card users.

Q19. Is this clustering unsupervised? Why?
A19. Yes. No target column is given; the algorithm finds groups only from the feature values.

Q20. What is truncate_mode="lastp"?
A20. A dendrogram option that shows only the last p merges (p=30). Smaller clusters are collapsed
     into numbered leaves (with the count in brackets), so the plot stays readable.

Q21. Why did the manual use PCA, and what does it cost?
A21. The credit card data has 17 features, which cannot be plotted. PCA reduces it to 2
     components so it can be clustered and drawn in 2-D, and it makes the dendrogram cheaper.
     The cost: the 2 components keep only 0.290 + 0.184 = 0.474, about 47% of the variance,
     so more than half of the information is dropped and the clusters describe only the 2-D
     view.

Q22. What is Euclidean distance?
A22. The straight line distance between two points: square root of the sum of squared
     differences of each feature.

Q23. Can we use categorical columns?
A23. Not directly: Euclidean distance needs numbers. A category could be encoded as 0 / 1, or a
     distance such as Gower distance used. All 17 columns here are numeric (CUST_ID, a text
     id, is dropped).

Q24. What does Ward minimise?
A24. At every step it merges the pair of clusters that increases the total within-cluster
     sum of squares the least (the same quantity K-Means tries to minimise).

Q25. How can you check the clusters are sensible without labels?
A25. Use the silhouette score, look at the scatter plot, compare linkage methods, and check the
     cluster profile (means of the original columns) for a meaningful business story.

Q26. Why fill missing values with the median instead of ffill?
A26. The manual's fillna(method='ffill') copies the value from the row above, which belongs to
     a different customer, so it invents a link between unrelated people; it also depends on row
     order and the method= argument is removed in current pandas. The median of the column is
     a typical value, is not pulled by the huge outliers in this data (MINIMUM_PAYMENTS goes up
     to 76,000), and works on all pandas versions. (Median fill gave silhouette 0.421 at k = 2,
     ffill about 0.432: a small difference.)

Q27. Why do we scale AND normalize?
A27. They do two different jobs. StandardScaler works on columns: it gives each feature mean 0
     and std 1 so no feature dominates because of its units. normalize works on rows: it makes
     each customer a unit-length vector so customers are compared by the shape of their
     behaviour, not by how big their numbers are. The manual does both in this order.

Q28. Why PCA before clustering?
A28. To get 2 dimensions that can be plotted and to make the distance work cheap and less
     noisy. The price is lost information (only about 47% of variance kept here).

Q29. Why do 8950 points make scipy linkage memory heavy?
A29. linkage() first builds all pairwise distances: n(n-1)/2 = about 40 million numbers for
     8950 points, around 320 MB, and the memory grows with the square of n. Doubling the number
     of customers makes it 4 times bigger, so hundreds of thousands of customers would not fit.
     That is why the dendrogram is truncated and why big data needs K-Means or sampling.

Q30. Why does silhouette pick k = 2 while the business may want more segments?
A30. Silhouette only measures how well separated the groups are, and on this data two broad
     groups (cash-advance users and shoppers) separate best (0.421), with the score falling as
     k grows (0.345 at k = 6). A marketing team may still want 4 or 5 smaller segments to
     target; the dendrogram also shows some structure below the top merge. The choice of k is a mix
     of the score and business usefulness.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Which line sets the number of clusters?
A1. AgglomerativeClustering(n_clusters=k) inside the loop for k in range(2, 7); the final
    profile uses best_k = ks[int(np.argmax(scores))].

Q2. Change the loop to range(2, 9). What changes?
A2. More k values are fitted, plotted (the 2 x 3 subplot grid must be made bigger) and scored;
    the bar chart gets more bars and the best k is picked again from the scores.

Q3. Change linkage="ward" to "single". What do you expect?
A3. Chaining: very uneven cluster sizes (often one big cluster) and a lower silhouette, as the
    comparison table shows (0.194).

Q4. Write the line that draws the full dendrogram without truncation.
A4. dendrogram(Z) (without truncate_mode and p). For 8950 points it is an unreadable black
    smudge and slow to draw.

Q5. How do you remove the normalize step?
A5. Use X_pca = pca.fit_transform(X_scaled) instead of X_normalized. The clusters then follow
    the size of the numbers too, so the result and the silhouette change.

Q6. How do you fill nulls the manual's way?
A6. df = df.ffill() (the modern form of fillna(method='ffill')). It copies the previous
    customer's value, which is why the median is preferred.

Q7. How do you read the first row of the 6-point merge table?
A7. It lists the ids of the two closest customers (0 and 4), the distance between them (0.211),
    and the size 2 of the new cluster, which gets id 6.

Q8. How do you print the average credit limit of every cluster?
A8. df.groupby("Cluster")["CREDIT_LIMIT"].mean().

Q9. How do you use a different distance such as Manhattan?
A9. AgglomerativeClustering(n_clusters=k, metric="manhattan", linkage="average"). Ward only
    works with Euclidean distance.

Q10. Why do we use fit_predict and not fit?
A10. fit_predict fits the model and returns the cluster number of each customer in one step.
     AgglomerativeClustering has no separate predict for new data.

Q11. How do you cut the scipy tree into 2 clusters directly?
A11. from scipy.cluster.hierarchy import fcluster; fcluster(Z, 2, criterion="maxclust").

Q12. How do you save the dendrogram image?
A12. plt.savefig("dendrogram.png") before plt.show().

Q13. How do you keep more of the information in PCA?
A13. Use more components, for example PCA(n_components=5) or PCA(n_components=0.9) (keep 90%
     of the variance). The clustering can then use all those columns, but the scatter plot can
     only show two of them.

Q14. How do you drop the id column?
A14. df = df.drop(columns="CUST_ID").

Q15. What would you check if the script runs out of memory?
A15. The number of points in linkage(): cluster a random sample, for example
     X_pca[np.random.choice(len(X_pca), 2000, replace=False)], or use K-Means for the full data.

END OF VIVA NOTES
'''
