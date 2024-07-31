# %%
# import libraries
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

# %%
# set general parameters
# corpus_file_name = "XXX"
rng = np.random.default_rng(seed=66)

# %%
# set similarity calculation parameters
window_size = 10
one_side_window_size = int(window_size / 2)

# %%
# set training paramters
frequency_cutoff = 5
iteration_n = 2  # XXX 1000
step_size = 0.01  # XXX 0.0001

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
training_word_n = len(training_words)
training_words.sort()
training_similarities = similarities.loc[training_words, training_words].to_numpy()
original_training_embeddings = np.array([embeddings[w] for w in training_words])
dimension_n = original_training_embeddings.shape[1]

# I haven't programmed testing yet XXX
# testing_word_n = len(testing_words)
# testing_words.sort()
# testing_similarities = similarities.loc[testing_words, testing_words].to_numpy()
# original_testing_embeddings = np.array([embeddings[w] for w in testing_words])
# steps:
# 1. use the final weights to transform testing_embeddings
# 2. calculate embedding similarities
# 3. compare to testing_similarities from co-occurrence data


# %%
# define functions for gradient descent
# I followed this: https://machinelearningmastery.com/gradient-descent-optimization-from-scratch/
def objective(weights):
    adjusted_embeddings = original_training_embeddings * weights
    embedding_similarities = cosine_similarity(adjusted_embeddings, adjusted_embeddings)
    return (
        adjusted_embeddings,
        embedding_similarities,
        mean_squared_error(training_similarities, embedding_similarities),
    )


def all_derivatives(weights, pred_sim, pred_embeddings):
    return np.array(
        [
            single_weight_derivative(i, weights, pred_sim, pred_embeddings)
            for i in range(len(weights))
        ]
    )


def single_weight_derivative(weight_i, weights, pred_sim, pred_embeddings):
    # modified from: https://sebastianraschka.com/faq/docs/mse-derivative.html
    w = weights[weight_i]
    n = weights.size
    obsv_sim = training_similarities.flatten()
    pred_sim = pred_sim.flatten()
    return np.array(
        [
            -2
            / n
            * (v - pred_sim[i])
            * single_weight_derivative_similarity(i, weight_i, w, pred_embeddings)
            for i, v in enumerate(obsv_sim)
        ]
    ).sum()


def pair_index_to_word_indices(pair_i):
    a = pair_i % training_word_n
    b = int((pair_i - a) / training_word_n)
    return (a, b)


def single_weight_derivative_similarity(pair_i, weight_i, weight, pred_embeddings):
    a, b = pair_index_to_word_indices(pair_i)
    vector_a = pred_embeddings[a, :]
    vector_b = pred_embeddings[b, :]
    e_ia = vector_a[weight_i]
    e_ib = vector_b[weight_i]
    l_a = np.linalg.norm(vector_a)
    l_b = np.linalg.norm(vector_b)

    # I did the calculus by hand:
    inner_product = np.inner(vector_a, vector_b)
    subterm = (e_ia**2) * (l_b**2) + (e_ib**2) * (l_a**2)
    coefficient = (2 * e_ia * e_ib / (l_a * l_b)) - (
        inner_product * subterm / ((l_a**3) * (l_b**3))
    )
    return coefficient * weight


def gradient_descent(objective, derivative, iteration_n, step_size, initial_values):
    weights = initial_values.copy()
    for i in range(iteration_n):
        pred_embeddings, pred_sim, _ = objective(weights)
        gradients = derivative(weights, pred_sim, pred_embeddings)
        weights = weights - step_size * gradients
        _, _, loss = objective(weights)
        print(f"{i}: loss({round(weights[0],6)},...) = {round(loss, 5)}")
    return [weights, loss]


# %%
# perform gradient descent
initial_weights = rng.random(dimension_n)
final_weights, final_loss = gradient_descent(
    objective, all_derivatives, iteration_n, step_size, initial_weights
)

# %%
