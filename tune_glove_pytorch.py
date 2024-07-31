# %%
# import libraries
import math
import re
from collections import defaultdict  # For word frequency

import numpy as np
import pandas as pd
from nltk.corpus import brown
import spacy
from gensim.models.phrases import Phrases, Phraser
import gensim.downloader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# %%
# set general parameters
# corpus_file_name = "XXX"
rng = np.random.default_rng(seed=99)

# %%
# set similarity calculation parameters
window_size = 10
one_side_window_size = int(window_size / 2)

# %%
# set training paramters
frequency_cutoff = 5
iteration_n = 30
initial_step_size = 200
decay_rate_k = 0.1

#   ######  ##       ########    ###    ##    ##
#  ##    ## ##       ##         ## ##   ###   ##
#  ##       ##       ##        ##   ##  ####  ##
#  ##       ##       ######   ##     ## ## ## ##
#  ##       ##       ##       ######### ##  ####
#  ##    ## ##       ##       ##     ## ##   ###
#   ######  ######## ######## ##     ## ##    ##
# %%
# load corpus
samples = [" ".join(brown.words(i)) for i in brown.fileids()]

# %%
# clean corpus: remove non-letter and stopwords and lemmatize corpus
# https://medium.com/@erkajalkumari/step-by-step-guide-to-word2vec-with-gensim-eb438d82bd01
samples = [re.sub(r"[^A-Za-z']+", " ", s).lower().strip(" ") for s in samples]
samples = [re.sub(r"\s+", " ", s) for s in samples]
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def cleaning(doc):
    return [token.lemma_ for token in doc if not token.is_stop]


samples = [cleaning(doc) for doc in nlp.pipe(samples, batch_size=5)]
samples = [[w for w in s if w not in ["'", "''"]] for s in samples]


#   ######  #### ##     ## #### ##          ###    ########  #### ######## ##    ##
#  ##    ##  ##  ###   ###  ##  ##         ## ##   ##     ##  ##     ##     ##  ##
#  ##        ##  #### ####  ##  ##        ##   ##  ##     ##  ##     ##      ####
#   ######   ##  ## ### ##  ##  ##       ##     ## ########   ##     ##       ##
#        ##  ##  ##     ##  ##  ##       ######### ##   ##    ##     ##       ##
#  ##    ##  ##  ##     ##  ##  ##       ##     ## ##    ##   ##     ##       ##
#   ######  #### ##     ## #### ######## ##     ## ##     ## ####    ##       ##
# %%
# calculate co-occurrence counts using decreasing weighting function (count += 1/dist)
def co_occurrence(samples, one_side_window_size):
    frequencies = defaultdict(int)
    cooccurrence_counts = defaultdict(int)
    for text in samples:
        for i in range(len(text)):
            this_word = text[i]
            frequencies[this_word] += 1
            # only one-side window needed because co-occurrence is non-directional
            # e.g., in 'hi there', (hi, there) = (there, hi) = 1
            for j in range(1, one_side_window_size + 1):
                if i + j < len(text):
                    key = tuple(sorted([this_word, text[i + j]]))
                    cooccurrence_counts[key] += 1 / j
    vocab = sorted(frequencies.keys())
    cooccurrences = pd.DataFrame(
        data=np.zeros((len(vocab), len(vocab))),
        index=vocab,
        columns=vocab,
    )
    for key, value in cooccurrence_counts.items():
        cooccurrences.loc[key[0], key[1]] = value
        cooccurrences.loc[key[1], key[0]] = value
    arr = cooccurrences.to_numpy()
    assert np.shares_memory(arr, cooccurrences)
    np.fill_diagonal(arr, 0)
    return cooccurrences, frequencies


cooccurrences, frequencies = co_occurrence(samples, one_side_window_size)

# %%
# # calculate co-occurrence probabilities
sums = cooccurrences.sum(axis=0)
cooccurrence_probabilities = cooccurrences.div(sums, axis=0)

# %%
# find frequent words
frequencies = {k: v for k, v in frequencies.items() if v >= frequency_cutoff}
words_by_freq = sorted(frequencies, key=frequencies.get, reverse=True)

# %%
# download GloVe embeddings
glove_vectors = gensim.downloader.load("glove-wiki-gigaword-300")

# %%
# find words that exist in GloVe
embeddings = {}
included_words = []
for w in words_by_freq:
    try:
        embeddings[w] = glove_vectors[w]
        included_words.append(w)
    except KeyError:
        pass
included_words_by_freq = included_words.copy()
included_words.sort()
frequencies = {k: v for k, v in frequencies.items() if k in included_words}


# %%
# remove infrequent words and words not in GloVe
cooccurrence_probabilities = cooccurrence_probabilities.loc[
    included_words, included_words
]

# %%
# calculate similarity matrix
cooccurrence_probabilities = cooccurrence_probabilities.to_numpy()
similarities = cosine_similarity(cooccurrence_probabilities, cooccurrence_probabilities)
similarities = pd.DataFrame(similarities, index=included_words, columns=included_words)

#  ######## ########     ###    #### ##    ##
#     ##    ##     ##   ## ##    ##  ###   ##
#     ##    ##     ##  ##   ##   ##  ####  ##
#     ##    ########  ##     ##  ##  ## ## ##
#     ##    ##   ##   #########  ##  ##  ####
#     ##    ##    ##  ##     ##  ##  ##   ###
#     ##    ##     ## ##     ## #### ##    ##
# %%
# split words into training and testing sets
word_n = len(included_words)
testing_words = included_words_by_freq[::10]
training_words = [w for w in included_words_by_freq if w not in testing_words]
training_words.sort()
testing_words.sort()
training_similarities = torch.tensor(
    similarities.loc[training_words, training_words].to_numpy()
)
original_training_embeddings = torch.tensor(
    np.array([embeddings[w] for w in training_words])
)
testing_similarities = similarities.loc[testing_words, testing_words].to_numpy()
original_testing_embeddings = np.array([embeddings[w] for w in testing_words])
dimension_n = original_training_embeddings.shape[1]


# %%
# define functions for gradient descent
def model(coefficients):
    adjusted_embeddings = original_training_embeddings * coefficients
    return pairwise_cos_sim(adjusted_embeddings)


def pairwise_cos_sim(a):
    numerator = torch.matmul(a, a.T)
    a_ss = torch.mul(a, a).sum(axis=1)
    denominator = torch.max(torch.sqrt(torch.outer(a_ss, a_ss)), torch.tensor(1e-8))
    return torch.div(numerator, denominator)


def loss_fn(pred, obsv):
    return torch.mean((pred - obsv) ** 2)


def exp_decay_step_size(t):
    return initial_step_size * math.exp(-decay_rate_k * t)


def test(coefficients):
    adjusted_embeddings = original_testing_embeddings * coefficients.detach().numpy()
    predicted_similarities = cosine_similarity(adjusted_embeddings, adjusted_embeddings)
    return mean_squared_error(predicted_similarities, testing_similarities)


# %%
# perform gradient descent
coefficients = torch.rand(dimension_n, requires_grad=True)
loss_records = []
performance_records = []
for i in range(iteration_n):
    predicted_similarities = model(coefficients)
    loss = loss_fn(predicted_similarities, training_similarities)
    loss.backward()
    with torch.no_grad():
        coefficients -= exp_decay_step_size(i) * coefficients.grad
        coefficients.grad.zero_()
    loss_records.append(loss.item())
    performance_records.append(test(coefficients))

with open("training_records.txt", "w") as f:
    loss_strings = [str(i) for i in loss_records]
    f.write("\n".join(loss_strings))

with open("performance_records.txt", "w") as f:
    performance_strings = [str(i) for i in performance_records]
    f.write("\n".join(performance_strings))

with open("fitted_coefficients.txt", "w") as f:
    coefficient_strings = [str(float(i.data)) for i in coefficients]
    f.write("\n".join(coefficient_strings))

# %%
# compare performance with original GloVe
glove_testing_similarities = cosine_similarity(
    original_testing_embeddings, original_testing_embeddings
)
baseline_performance = mean_squared_error(
    glove_testing_similarities, testing_similarities
)

# %%
# plot training process
small_text_size = 10
big_text_size = 15
rc("font", **{"family": "sans-serif", "sans-serif": ["Century Gothic"]})
plot_settings = {"axes.edgecolor": "gray", "lines.solid_capstyle": "butt"}

records = pd.DataFrame(columns=["loss", "performance"], index=range(1, iteration_n + 1))
records["loss"] = loss_records
records["performance"] = performance_records

with sns.axes_style("white", rc=plot_settings):
    f, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(data=records, ax=ax)
    ax.axhline(y=baseline_performance, color="green", linestyle="--", label="GloVe")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)
    ax.set(ylim=(0.009, 0.014))
    plt.ylabel("Mean Squared Error", fontsize=big_text_size, labelpad=5)
    plt.xlabel("Iteration", fontsize=big_text_size, labelpad=6)
    plt.xticks(range(0, iteration_n + 1, 10), fontsize=small_text_size)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend()
    # plt.legend(labels=["loss", "performance", "GloVe"])
    f.savefig("results.png", dpi=200, bbox_inches="tight", transparent=False)

# %%
