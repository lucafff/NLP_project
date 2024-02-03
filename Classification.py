from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from networkx import DiGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import copy

from utils import read_json_file
from utils import fit_classifier
from evaluation import evaluate_h

# Construct the graph
G = DiGraph()
G.add_edge(ROOT, "Free")
G.add_edge(ROOT, "Persuasion")
G.add_edge("Persuasion", "Logos")
G.add_edge("Logos", "Repetition")
G.add_edge("Logos", "Obfuscation, Intentional vagueness, Confusion")
G.add_edge("Logos", "Reasoning")
G.add_edge("Logos", "Justification")
G.add_edge('Justification', "Slogans")
G.add_edge('Justification', "Bandwagon")
G.add_edge('Justification', "Appeal to authority")
G.add_edge('Justification', "Flag-waving")
G.add_edge('Justification', "Appeal to fear/prejudice")
G.add_edge('Reasoning', "Simplification")
G.add_edge('Simplification', "Causal Oversimplification")
G.add_edge('Simplification', "Black-and-white Fallacy/Dictatorship")
G.add_edge('Simplification', "Thought-terminating cliché")
G.add_edge('Reasoning', "Distraction")
G.add_edge('Distraction', "Misrepresentation of Someone's Position (Straw Man)")
G.add_edge('Distraction', "Presenting Irrelevant Data (Red Herring)")
G.add_edge('Distraction', "Whataboutism")
G.add_edge("Persuasion", "Ethos")
G.add_edge('Ethos', "Appeal to authority")
G.add_edge('Ethos', "Glittering generalities (Virtue)")
G.add_edge('Ethos', "Bandwagon")
G.add_edge('Ethos', "Ad Hominem")
G.add_edge('Ad Hominem', "Doubt")
G.add_edge('Ad Hominem', "Name calling/Labeling")
G.add_edge('Ad Hominem', "Smears")
G.add_edge('Ad Hominem', "Reductio ad hitlerum")
G.add_edge('Ad Hominem', "Whataboutism")
G.add_edge("Persuasion", "Pathos")
G.add_edge('Pathos', "Exaggeration/Minimisation")
G.add_edge('Pathos', "Loaded Language")
G.add_edge('Pathos', "Appeal to fear/prejudice")
G.add_edge('Pathos', "Flag-waving")





train_path = "./data/train.json"
validation_path = "./data/validation.json"
test_path = "./data/dev_subtask1_en.json"

if __name__ == '__main__':
    # Leggi train.json
    train_data = read_json_file(train_path)

    # Leggi validation.json
    validation_data = read_json_file(validation_path)

    test_data = read_json_file(test_path)


    # Inizializza le liste per i dati di addestramento
    texts_train, labels_train = [], []
    texts_validation, labels_validation = [], []
    texts_test, labels_test = [], []
    texts_test_vero, labels_test_vero = [], []
    if train_data:
        texts_train = [element['text'] for element in train_data]
        labels_train = [element['labels'] if element['labels'] else ['Free'] for element in train_data]

    # Estre x e y dai dataset se esistono
    if validation_data:
        texts_validation = [element['text'] for element in validation_data]
        labels_validation = [element['labels'] if element['labels'] else ['Free'] for element in validation_data]

    if test_data:
        texts_test = [element['text'] for element in test_data]
        labels_test = [element['labels'] if element['labels'] else ['Free'] for element in test_data]

    # Unisce i dati di addestramento e validation
    texts = texts_train + texts_validation
    labels = labels_train + labels_validation


    """
    # Divide dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)  # , stratify=results
    """
    ####### nuova sezione per avere validation.json come gold
    X_test = texts_test
    y_test = labels_test
    X_train = texts
    y_train = labels


    vectorizer = TfidfVectorizer()
    X_test = vectorizer.fit_transform(X_test)

    # Definizione dei classificatori
    Free_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    Persuasion_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    Ethos_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    Ad_hominem_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    Pathos_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    Logos_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    Justification_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    Reasoning_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    Distraction_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    Simplification_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

    #addestramento dei classificatori tramite apposita funzione che gli permette di usare solo il sottoinsieme del dataset che gli serve
    Free_classifier = fit_classifier(X_train, y_train, G, ["Free", "Persuasion"], Free_classifier, vectorizer)
    Persuasion_classifier = fit_classifier(X_train, y_train, G, ["Ethos", "Pathos", "Logos"], Persuasion_classifier, vectorizer)
    Ethos_classifier = fit_classifier(X_train, y_train, G, ["Ad Hominem", "Bandwagon", "Glittering generalities (Virtue)" , "Appeal to authority"], Ethos_classifier, vectorizer)
    Ad_hominem_classifier = fit_classifier(X_train, y_train, G, ["Doubt", "Name calling/Labeling", "Smears", "Reductio ad hitlerum","Whataboutism"], Ad_hominem_classifier, vectorizer)
    Pathos_classifier = fit_classifier(X_train, y_train, G, ["Exaggeration/Minimisation", "Loaded Language", "Appeal to fear/prejudice", "Flag-waving"], Pathos_classifier, vectorizer)
    Logos_classifier = fit_classifier(X_train, y_train, G, ["Repetition", "Obfuscation, Intentional vagueness, Confusion", "Reasoning", "Justification"], Logos_classifier, vectorizer)
    Justification_classifier = fit_classifier(X_train, y_train, G, ["Slogans", "Bandwagon", "Appeal to authority", "Flag-waving", "Appeal to fear/prejudice"], Justification_classifier, vectorizer)
    Reasoning_classifier = fit_classifier(X_train, y_train, G, ["Simplification", "Distraction"], Reasoning_classifier, vectorizer)
    Distraction_classifier = fit_classifier(X_train, y_train, G, ["Whataboutism","Misrepresentation of Someone's Position (Straw Man)","Presenting Irrelevant Data (Red Herring)"], Distraction_classifier, vectorizer)
    Simplification_classifier = fit_classifier(X_train, y_train, G, ["Causal Oversimplification", "Black-and-white Fallacy/Dictatorship", "Thought-terminating cliché"], Simplification_classifier, vectorizer)

    y_pred = []

    for i in range(X_test.shape[0]):
        y_pred.append([])

        #probabilities = Free_classifier.predict_proba(X_test[i])
        prediction = Free_classifier.predict(X_test[i])

        # Free
        if prediction[0, 0] == 1:
            #y_pred[i].append("Free")
            continue

        # Persuasion
        if prediction[0, 1] == 1:
            prediction = Persuasion_classifier.predict(X_test[i])

            #Ethos
            if prediction[0, 0] == 1:
                prediction1 = Ethos_classifier.predict(X_test[i])

                # Ad Hominem
                if prediction1[0, 0] == 1:
                    prediction8 = Ad_hominem_classifier.predict(X_test[i])
                    if prediction8[0, 0] == 1:
                        y_pred[i].append("Doubt")
                    if prediction8[0, 1] == 1:
                        y_pred[i].append("Name calling/Labeling")
                    if prediction8[0, 2] == 1:
                        y_pred[i].append("Smears")
                    if prediction8[0, 3] == 1:
                        y_pred[i].append("Reductio ad hitlerum")
                    if prediction8[0, 4] == 1:
                        y_pred[i].append("Whataboutism")
                    if all(prediction8[0, i] == 0 for i in range(5)):
                        y_pred[i].append("Ad Hominem")

                if prediction1[0, 1] == 1:
                    y_pred[i].append("Bandwagon")
                if prediction1[0, 2] == 1:
                    y_pred[i].append("Glittering generalities (Virtue)")
                if prediction1[0, 3] == 1:
                    y_pred[i].append("Appeal to authority")
                if all(prediction1[0, i] == 0 for i in range(4)):
                    y_pred[i].append("Ethos")
            #Pathos
            if prediction[0, 1] == 1:
                prediction2 = Pathos_classifier.predict(X_test[i])
                if prediction2[0, 0] == 1:
                    y_pred[i].append("Exaggeration/Minimisation")
                if prediction2[0, 1] == 1:
                    y_pred[i].append("Loaded Language")
                if prediction2[0, 2] == 1:
                    y_pred[i].append("Appeal to fear/prejudice")
                if prediction2[0, 3] == 1:
                    y_pred[i].append("Flag-waving")
                if all(prediction2[0, i] == 0 for i in range(4)):
                    y_pred[i].append("Pathos")
            #Logos
            if prediction[0, 2] == 1:
                prediction3 = Logos_classifier.predict(X_test[i])
                if prediction3[0, 0] == 1:
                    y_pred[i].append("Repetition")
                if prediction3[0, 1] == 1:
                    y_pred[i].append("Obfuscation, Intentional vagueness, Confusion")

                # Reasoning
                if prediction3[0, 2] == 1:
                    prediction4 = Reasoning_classifier.predict(X_test[i])

                    #Simplification
                    if prediction4[0, 0] == 1:
                        prediction6 = Simplification_classifier.predict(X_test[i])
                        if prediction6[0, 0] == 1:
                            y_pred[i].append("Causal Oversimplification")
                        if prediction6[0, 1] == 1:
                            y_pred[i].append("Black-and-white Fallacy/Dictatorship")
                        if prediction6[0, 2] == 1:
                            y_pred[i].append("Thought-terminating cliché")
                        if all(prediction6[0, i] == 0 for i in range(3)):
                            y_pred[i].append("Simplification")

                    #Distraction
                    if prediction4[0, 1] == 1:
                        prediction7 = Distraction_classifier.predict(X_test[i])
                        if prediction7[0, 0] == 1:
                            y_pred[i].append("Whataboutism")
                        if prediction7[0, 1] == 1:
                            y_pred[i].append("Misrepresentation of Someone's Position (Straw Man)")
                        if prediction7[0, 2] == 1:
                            y_pred[i].append("Presenting Irrelevant Data (Red Herring)")
                        if all(prediction7[0, i] == 0 for i in range(3)):
                            y_pred[i].append("Distraction")

                #Justification
                if prediction3[0, 3] == 1:
                    prediction5 = Justification_classifier.predict(X_test[i])
                    if prediction5[0, 0] == 1:
                        y_pred[i].append("Slogans")
                    if prediction5[0, 1] == 1:
                        y_pred[i].append("Bandwagon")
                    if prediction5[0, 2] == 1:
                        y_pred[i].append("Appeal to authority")
                    if prediction5[0, 3] == 1:
                        y_pred[i].append("Flag-waving")
                    if prediction5[0, 4] == 1:
                        y_pred[i].append("Appeal to fear/prejudice")
                    if all(prediction5[0, i] == 0 for i in range(5)):
                        y_pred[i].append("Justification")

                if all(prediction3[0, i] == 0 for i in range(4)):
                    y_pred[i].append("Logos")

            #if all(prediction[0, i] == 0 for i in range(3)):
            #    y_pred[i].append("Persuasion")


    # Per evitare ripetizioni delle foglie raggiunguibili da più percorsi
    for i in range(len(y_pred)):
        y_pred[i] = list(set(y_pred[i]))




# Creazione di copie profonde per evitare riferimenti condivisi
validation_data_aux = copy.deepcopy(test_data)

# Sostituisci i valori del campo 'labels' di validation_data_aux con quelli di aux
for i, element in enumerate(validation_data_aux):
    element["labels"] = y_pred[i]


# Salva il nuovo validation_data modificato in un file pred.json
output_file_path = "./data/pred.json"
with open(output_file_path, "w", encoding='utf-8') as output_file:
    json.dump(validation_data_aux, output_file, indent=2)



prec_h, rec_h, f1_h = evaluate_h("./data/pred.json", "./data/dev_subtask1_en.json", G)
print("f1_h={:.5f}\tprec_h={:.5f}\trec_h={:.5f}".format(f1_h, prec_h, rec_h))




















