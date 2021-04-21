import pandas as pd
import numpy as np
from util import log, timeclass


class Preprocessor:
    def __init__(self):
        pass
    
    def fit(self,ss):
        pass
    
    def transform(self,ss):
        pass

    def fit_transform(self,ss):
        pass

class GeneralPreprocessor(Preprocessor):
    def __init__(self,X):
        self.X=X
    
    def transform(self):
        self.userColRefine()
        self.selectApartCat()

    def selectApartCat(self):
        '''
        筛选类别列，把类别区分度不高的列删掉
        '''
        df = X.data
        apart_cols = []

        def func(ss):
            length = len(ss)//100
            part1 = ss.iloc[:length*49].dropna().drop_duplicates()
            part2 = ss.iloc[length*49:].dropna().drop_duplicates()
            union_len = len(pd.concat([part1,part2]).drop_duplicates())
            inter_len = len(part1) + len(part2) - union_len
            if union_len == 0:
                return True

            if inter_len/union_len <= 0.001:
                return True
            else:
                return False

        todo_cols = X.session_cols+X.user_cols+X.key_cols+X.cat_cols
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[col]) for col in todo_cols)

        for col,apart in zip(todo_cols,res):
            if apart:
                apart_cols.append(col)

        self.apart_cat_cols = apart_cols

        X.add_apart_cat_cols(self.apart_cat_cols)
        X.add_post_drop_cols(self.apart_cat_cols)
        log(f'apart_cat_cols:{self.apart_cat_cols}')

        
    def userColRefine(self):
        '''
        给用户列重新编码
        '''
        if len(self.X.session_cols)!=0 or not self.X.user_cols or not self.X.key_time_col:
            return

        df = self.X.data.sort_index()

        user_col = df[self.X.user_cols[0]]
        ss = user_col.diff()

        time_col = df[self.X.key_time_col]
        time_diff = time_col.diff().dt.total_seconds()

        judge = ((ss!=0) | (time_diff>3600) | (time_diff<-3600))

        unique = ss[judge].shape[0]
        ss.loc[judge] = [i+1 for i in range(unique)]
        ss.loc[~judge] = np.nan
        ss = ss.fillna(method='ffill')
        new_col = 'userColRefine'
        new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.CATEGORY_TYPE)
        ss.name = new_col

        df = self.X.data
        df[new_col] = ss
        self.X.update_data(df,{},None)
        self.X.add_session_col(new_col)
        self.X.add_apart_cat_cols([new_col])
        self.X.add_post_drop_cols([new_col])