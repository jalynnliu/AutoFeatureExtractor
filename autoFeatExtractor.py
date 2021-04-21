import os

import numpy as np
import pandas as pd
import copy
from datetime import datetime
import gc
import time

from util import log, timeclass

class AFE:
    auc = []
    ensemble_auc = []
    ensemble_train_auc = []

    def __init__(self, info):
        self.info = copy.deepcopy(info)
        self.tables = None

    def shuffle(self, X, y, random_state):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X.iloc[idx]
        y = y.iloc[idx]
        return X, y

    def release_tables(self, Xs, graph):

        for name in graph.tables:
            del Xs[name]
            del graph.name2table[name]

        gc.collect()

    @timeclass(cls='AFE')
    def process(self, Xs, y, X_test):
        np.random.seed(CONSTANT.SEED)

        Xs[Config.MAIN_TABLE_NAME] = pd.concat(
            [Xs[Config.MAIN_TABLE_NAME], ])

        gc.collect()

        graph = Graph(self.info, Xs)
        # TODO merge tables, DFS

        feat_pipeline = FeatPipeline()
        feat_engine = FeatEngine(feat_pipeline, config)
        feat_engine.fit_transform_order1(main_table, y)
        # TODO Feature genetator

        def split_table(table, y):
            X = table.data
            X_train, y_train, X_test, y_test = train_test_split(
                X, y, shuffle=False, test_rate=0.2)
            table1 = copy.deepcopy(table)
            table1.data = X_train
            table2 = copy.deepcopy(table)
            table2.data = X_test
            return table1, y_train, table2, y_test

        table1, y_train, table2, y_test = split_table(main_table, y)
        # TODO feature evaluation

        paths = os.path.join(feature_importance_path, version)
        if not os.path.exists(paths):
            os.makedirs(paths)
        importances.to_csv(os.path.join(paths, '{}_importances.csv'.format(
            datetime.now().strftime('%Y%m%d%H%M%S'))), index=False)


    @timeclass(cls='AFE')
    def fit(self, Xs, y):
        self.Xs = Xs
        self.y = y
        