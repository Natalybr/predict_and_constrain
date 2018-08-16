import numpy as np
from sklearn.metrics import f1_score
import arff  # pip install liac-arff
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def evaluate_f1(predictor, features, labels):
    """Compute the F1 performance of a predictor on the given data."""
    mean_f = []

    for idx, (feature, lbl) in enumerate(zip(features, labels)):
        pred_lbl = predictor(feature)

        f1 = f1_score(lbl, pred_lbl)
        mean_f.append(f1)
        if idx % 100 == 0:
            print "%.3f (%d of %d)" % (np.mean(mean_f), idx, len(features))
    print "%.3f" % (np.mean(mean_f))
    return np.mean(mean_f)


def get_bibtex(split='train'):
    assert split in ['train', 'test']
    feature_idx = 1836
    if split == 'test':
        dataset = arff.load(open('%s/bibtex/bibtex-test.arff' % dir_path, 'rb'))
    else:
        dataset = arff.load(open('%s/bibtex/bibtex-train.arff' % dir_path, 'rb'))

    data = np.array(dataset['data'], np.int)

    labels = data[:, feature_idx:]
    features = data[:, 0:feature_idx]
    txt_labels = [t[0] for t in dataset['attributes'][1836:]]
    txt_inputs = [t[0] for t in dataset['attributes'][:1836]]

    if split == 'train':
        return labels, features, txt_labels
    else:
        return labels, features, txt_labels, txt_inputs
