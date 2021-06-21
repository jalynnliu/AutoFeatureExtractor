
import time
import functools
import numpy as np
from typing import Any

from . import CONSTANT

nesting_level = 0
is_start = None


class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)


def timeclass(cls):
    def timeit(method, start_log=None):
        @functools.wraps(method)
        def timed(*args, **kw):
            global is_start
            global nesting_level
    
            if not is_start:
                print()
    
            is_start = True
            log(f"Start [{cls}.{method.__name__}]:" + (start_log if start_log else ""))
            log(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            nesting_level += 1
    
            start_time = time.time()
            result = method(*args, **kw)
            end_time = time.time()
    
            nesting_level -= 1
            log(f"End   [{cls}.{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
            log(f'End time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            is_start = False
    
            return result
    
        return timed
    return timeit

def timeit(method, start_log=None):
    @functools.wraps(method)
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

      

class FeatContext:
    @staticmethod
    def gen_feat_name(namespace,cls_name,feat_name,feat_type):
        prefix = CONSTANT.type2prefix[feat_type]


        return f"{prefix}{cls_name}:{feat_name}:{namespace}"

    @staticmethod
    def gen_merge_name(table_name,feat_name,feat_type):
        prefix = CONSTANT.type2prefix[feat_type]
        return f"{prefix}{table_name}.({feat_name})"

    @staticmethod
    def gen_merge_feat_name(namespace,cls_name,feat_name,feat_type,table_name):
        feat_name = FeatContext.gen_feat_name(namespace,cls_name,feat_name,feat_type)
        return FeatContext.gen_merge_name(table_name,feat_name,feat_type)


def train_test_split(X,y,test_rate=0.2,shuffle=True,random_state=1):
    length = X.shape[0]


    test_size = int(length * test_rate)
    train_size = length - test_size

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    if shuffle:
        np.random.seed(random_state)
        idx = np.arange(train_size)
        np.random.shuffle(idx)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]

    return X_train,y_train,X_test,y_test