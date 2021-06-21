import lightgbm as lgb
import numpy as np
from utils import CONSTANT
from utils.util import log, timeclass
import pandas as pd
import gc
from . import autosample
import time
import copy

class AutoLGB():
    def __init__(self):
        self.params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "verbosity": 1,
            "seed": CONSTANT.SEED,
            "num_threads": CONSTANT.THREAD_NUM
        }

        self.hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin':255,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

        self.early_stopping_rounds = 50

    @timeclass(cls='AutoLGB')
    def predict(self,X):
        X = X[self.columns]
        X.columns = self.new_feat_name_cols
        return self.model.predict(X)

    @timeclass(cls='AutoLGB')
    def ensemble_train(self,X,y,categories,config,len_test):
        feat_name = list(X.columns)
        self.ensemble_models = []
        self.ensemble_columns = []
        columns = list(X.columns)
        log(f'lgb training set shape: {X.shape}')
        pos = (y==1).sum()
        neg = (y==0).sum()
        log(f'pos {pos} neg {neg}')

        self.columns = columns
        max_sample_num = len(y)

        feat_name_cols = list(X.columns)
        feat_name_maps = { feat_name_cols[i] : str(i)  for i in range(len(feat_name_cols)) }
        f_feat_name_maps = { str(i) : feat_name_cols[i] for i in range(len(feat_name_cols)) }
        new_feat_name_cols = [ feat_name_maps[i] for i in feat_name_cols ]
        X.columns = new_feat_name_cols
        categories = [ feat_name_maps[i] for i in categories ]
        self.f_feat_name_maps = f_feat_name_maps
        self.new_feat_name_cols = new_feat_name_cols
        
        all_columns = list(X.columns)
        
        start_time = time.time()
        i = 0
        cur_columns = all_columns
        seed = np.random.randint(2019*i,2019*(i+1))
        X_train,y_train = autosample.downsampling(X,y,max_sample_num,seed)
        X_train = X_train[cur_columns]
        gc.collect()
        
        colset = set(X_train.columns)
        cur_categorical = [col for col in categories if col in colset]
        pos = (y_train==1).sum()
        neg = (y_train==0).sum()

        params = self.params
        hyperparams = self.hyperparams
        params['seed'] = seed
        
        X_train = X_train.astype(np.float32)
        gc.collect()
        y_train = y_train.astype(np.float32)
        gc.collect()
        X_train = X_train.values
        gc.collect()
        y_train = y_train.values
        gc.collect()
        
        train_data = lgb.Dataset(X_train, label=y_train,feature_name=feat_name)
        del X_train,y_train
        gc.collect()
        
        model = lgb.train({**params, **hyperparams},
                                train_data,
                                num_boost_round=self.best_iteration,
                                feature_name=cur_columns,
                                categorical_feature=cur_categorical,
                                learning_rates = self.learning_rates[:self.best_iteration])

        self.ensemble_columns.append(cur_columns)
        self.ensemble_models.append(model)
        end_time = time.time()
        
        model_use_time = end_time - start_time
        del train_data
        
        gc.collect()
        
        start_time = time.time()
        temp = X.iloc[:100000]
        
        temp = temp.astype(np.float32)
        gc.collect()
        temp = temp.values
        gc.collect()
        
        model.predict(temp)
        
        end_time = time.time()
        model_test_use_time = (end_time-start_time)
        model_test_use_time = len_test/temp.shape[0] * model_test_use_time
        model_use_time = model_use_time + model_test_use_time
        del temp,model
        
        rest_time = config.budget/10*9-(end_time-config.start_time)
        if rest_time <= 0:
            rest_model_num = 0
        else:
            rest_model_num = int(rest_time // model_use_time)
        
        if rest_model_num >= 50:
            rest_model_num = 50 
            
        if rest_model_num >= 1:
            rest_model_num -= 1

        if not CONSTANT.USE_ENSEMBLE:
            rest_model_num = 0
        
        for i in range(1,rest_model_num+1):

            seed = np.random.randint(2019*i,2019*(i+1))
            
            cur_columns = list(pd.Series(all_columns).sample(frac=0.85,replace=False,random_state=seed))

            X_train,y_train = autosample.downsampling(X,y,max_sample_num,seed)
            X_train = X_train[cur_columns]
            gc.collect()
            
            colset = set(X_train.columns)
            cur_categorical = [col for col in categories if col in colset]

            pos = (y_train==1).sum()
            neg = (y_train==0).sum()

            params = self.params
            hyperparams = self.hyperparams
            params['seed'] = seed
            
            num_leaves = hyperparams['num_leaves']
            num_leaves = num_leaves + np.random.randint(-int(num_leaves/10),int(num_leaves/10)+7)
            
            lrs = np.array(self.learning_rates)
            rands = 1 + 0.2*np.random.rand(len(lrs))
            lrs = list(lrs * rands)
            
            cur_iteration = self.best_iteration
            cur_iteration = cur_iteration + np.random.randint(-30,40)
            if cur_iteration > len(lrs):
                cur_iteration = len(lrs)
            
            if cur_iteration <= 10:
                cur_iteration = self.best_iteration
            
            cur_hyperparams = copy.deepcopy(hyperparams)
            cur_hyperparams['num_leaves'] = num_leaves
            
            X_train = X_train.astype(np.float32)
            gc.collect()
            y_train = y_train.astype(np.float32)
            gc.collect()
            X_train = X_train.values
            gc.collect()
            y_train = y_train.values
            gc.collect()
            
            train_data = lgb.Dataset(X_train, label=y_train,feature_name=cur_columns)
            del X_train,y_train
            gc.collect()
            
            model = lgb.train({**params, **cur_hyperparams},
                                    train_data,
                                    num_boost_round=cur_iteration,
                                    feature_name=cur_columns,
                                    categorical_feature=cur_categorical,
                                    learning_rates = lrs[:cur_iteration])


            self.ensemble_columns.append(cur_columns)
            self.ensemble_models.append(model)

            del train_data
            gc.collect()

        X.columns = self.columns