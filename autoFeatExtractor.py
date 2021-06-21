import os

import numpy as np
import pandas as pd
import copy
from datetime import datetime
import gc

from utils import CONSTANT
from utils.util import timeclass, train_test_split
from feat.feat_pipe import FeatPipeline,FeatEngine

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

        Xs[CONSTANT.MAIN_TABLE_NAME] = pd.concat(
            [Xs[CONSTANT.MAIN_TABLE_NAME], ])

        gc.collect()

        feat_pipeline = FeatPipeline()
        feat_engine = FeatEngine(feat_pipeline, config)
        feat_engine.fit_transform_order1(Xs, y)
        feat_engine.fit_transform_keys_order2(Xs,y)
        del feat_engine
        gc.collect()


        def split_table(table, y):
            X = table.data
            X_train, y_train, X_test, y_test = train_test_split(
                X, y, shuffle=False, test_rate=0.2)
            table1 = copy.deepcopy(table)
            table1.data = X_train
            table2 = copy.deepcopy(table)
            table2.data = X_test
            return table1, y_train, table2, y_test

        table1, y_train, table2, y_test = split_table(Xs, y)
        feat_engine = FeatEngine(feat_pipeline,config)
        feat_output = FeatOutput()

        feat_engine.fit_transform_merge_order1(table1,y_train)
        X_train,y_train,categories = feat_output.fit_transform_output(table1,y_train)
        gc.collect()
        feat_engine.transform_merge_order1(table2)
        X_test = feat_output.transform_output(table2)

        lgb = AutoLGB()
        lgb.param_compute(X_train,y_train,categories,config)
        lgb.param_opt_new(X_train,y_train,X_test,y_test,categories)
        len_test = X_test.shape[0]
        lgb.ensemble_train(X_train,y_train,categories,config,len_test)
        gc.collect()

        pred,pred0 = lgb.ensemble_predict_test(X_test)

        auc = roc_auc_score(y_test,pred0)
        print('source AUC:',auc)
            
        auc = roc_auc_score(y_test,pred)
        Model.ensemble_auc.append(auc)
        print('ensemble AUC:',auc)
            
        importances = lgb.get_ensemble_importances()

        paths = os.path.join(feature_importance_path, version)
        if not os.path.exists(paths):
            os.makedirs(paths)
        importances.to_csv(os.path.join(paths, '{}_importances.csv'.format(
            datetime.now().strftime('%Y%m%d%H%M%S'))), index=False)


    @timeclass(cls='AFE')
    def fit(self, Xs, y):
        self.Xs = Xs
        self.y = y
        