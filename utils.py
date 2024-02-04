import os
import json
import numpy as np


# serie di 4 funzioni per la suddivisione del dataset secondo etichette
def find_descendants(graph, labels):
    descendants = set()

    def dfs(node):
        descendants.add(node)
        [dfs(child) for child in graph.successors(node)]

    [dfs(label) for label in labels]
    return descendants

def filter_matrix_by_labels(X, y, labels, graph):
    descendants_set = find_descendants(graph, labels)

    filtered_X, filtered_labels = zip(*[(X[i], list(set(labels_list) & descendants_set)) for i, labels_list in enumerate(y) if set(labels_list) & descendants_set])

    return np.vstack(filtered_X), list(filtered_labels)

def find_ancestor_in_labels(graph, node, target_labels):
    ancestors = set()

    def dfs(current):
        ancestors.add(current)
        [dfs(pred) for pred in graph.predecessors(current)]

    [dfs(predecessor) for predecessor in graph.predecessors(node)]

    common_ancestors = ancestors & set(target_labels)
    return list(common_ancestors)[0] if common_ancestors else None

def transform_Y_with_ancestors(graph, Y, target_labels):
    return [list(set([find_ancestor_in_labels(graph, label, target_labels) or label for label in labels_list])) for labels_list in Y]
###############################


def fit_classifier(X_Train, y_Train, G, labels_to_check, classifier, vectorizer):
    X, Y = filter_matrix_by_labels(X_Train, y_Train, labels_to_check, G)
    new_Y = transform_Y_with_ancestors(G, Y, labels_to_check)
    Y = Binarizzatore_etichette(labels_to_check, new_Y)
    X_tfidf = vectorizer.transform(X.flatten().tolist())

    return classifier.fit(X_tfidf, Y)


def Binarizzatore_etichette(lista_stringhe, lista_liste):
    # Inizializza la matrice con zeri
    matrice = [[0] * len(lista_stringhe) for _ in range(len(lista_liste))]

    # Riempie la matrice con 1 dove una stringa è presente nella lista di liste
    for i, lista in enumerate(lista_liste):
        for j, stringa in enumerate(lista_stringhe):
            if stringa in lista:
                matrice[i][j] = 1

    return matrice


def read_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    else:
        raise FileNotFoundError(f"File in '{file_path}' not found.")
    

def predict_check_status(i, duration):
    if i == 0:
        print("0%", end='', flush=True)
    elif i == duration/4:
        print('\b\b25%', end='', flush=True)
    elif i == duration/2:
        print('\b\b\b50%', end='', flush=True)
    elif i == (duration/4)*3: 
        print('\b\b\b75%', end='', flush=True)
    elif i == duration-1: 
        print('\b\b\b100%\n', flush=True)



