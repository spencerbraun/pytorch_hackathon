import re

import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

MODEL_PATH = 'all_data2_model.dbow_numnoisewords.2_vecdim.100_batchsize.32_lr.0.001000_epoch.95_loss.0.6850212.csv'


def readDocMatrix(path):
    input_lines = []
    with open('all_data2.csv') as f:
        for line in f:
            input_lines.append(line)

    output_lines = []
    with open(path) as f:
        for line in f:
            output_lines.append(line)

    matched = []
    for line_idx in range(len(output_lines)):
        matched.append((input_lines[line_idx], output_lines[line_idx]))
        
    matched_clean = [x for x in matched if not re.match(".*[A-Z].*", x[1])]

    return matched_clean


def toNumpyMatrix(matched):

    matched_clean_vectors = [x[1].split(",") for x in matched[1:]]
    vec_matrix = np.array(matched_clean_vectors)

    return vec_matrix


def topNMatches(idx, matched, n=5):
    
    matrix = toNumpyMatrix(matched)
    distances = distance.cdist([matrix[idx]], matrix, "cosine")[0]

    ind = np.argpartition(distances, n+1)[:(n+1)]
    sorted_ind = ind[np.argsort(distances[ind])][1:]
    min_distances = distances[sorted_ind]
    max_similarity = [1 - x for x in min_distances]
    
    return (max_similarity, sorted_ind)


def tsneMatrix(matched):

    doc_matrix = toNumpyMatrix(matched)
    doc_pca = PCA(n_components=5).fit_transform(doc_matrix)
    tsne = TSNE(n_components=2, perplexity=5).fit_transform(doc_pca)
    
    return tsne


def processTable(table, idx):
    probs = ["{:.2f}".format(x * 100) for x in table.iloc[idx].Rec_Probs]
    locs = table.iloc[idx].Rec_Index

    outputTable = (
        table
        .loc[locs]
        .join(pd.DataFrame(probs, index=locs, columns=['Similarity Score']))
    )
    
    return outputTable