#!/usr/bin/env python

import model
import mlc_datasets


def run_bibtex():
    # Get training data and build model
    train_labels, train_features, _ = mlc_datasets.get_bibtex('train')
    net = model.Model('./bibtex_model',
                      feature_dim=train_features.shape[1],
                      label_dim=train_labels.shape[1],
                      learning_rate=0.1,
                      inf_lr=0.005,
                      num_hidden=150,
                      weight_decay=0.001)

    # Train the model on the training data (first with validation, then over all training data)
    net.train_model(train_features, train_labels, train_ratio=0.95, epochs=20)
    net.train_model(train_features, train_labels, train_ratio=1, epochs=2)

    # Evaluate the final model on test data
    test_labels, test_features, _, __ = mlc_datasets.get_bibtex('test')
    mlc_datasets.evaluate_f1(net.predict, test_features, test_labels)

    return net


if __name__ == '__main__':
    run_bibtex()
