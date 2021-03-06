import gc
import pandas as pd
from joblib import Parallel, delayed

from utils import CONSTANT
from utils.util import timeclass,FeatContext
from utils.data_tool import gen_combine_cats,downcast

namespace = 'default'

class Feat:
    def __init__(self,config):
        self.config = config
    
    def fit(self,X,y):
        pass

    def transform(self,X):
        pass
    
    def fit_transform(self,X,y):
        pass

class CatNumStatistic(Feat):
    @timeclass(cls='CatNumStatistic')
    def fit(self,x,y):
        pass

    @timeclass(cls='CatNumStatistic')
    def transform(self, X):
        df=X.data

        for cat_col in X.cat_cols:
            for num_col in X.num_cols:
                df.groupby(cat_col)[num_col].mean()
                df.groupby(cat_col)[num_col].max()
                df.groupby(cat_col)[num_col].min()
                df.groupby(cat_col)[num_col].std()


    
class KeysNumStd(Feat):
    @timeclass(cls='KeysNumStd')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysNumStd')
    def transform(self,X,useful_cols=None):
        df = X.data

        todo_cols = X.user_cols+X.key_cols+X.session_cols
        num_cols = X.combine_num_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_std]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_std'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)

        else:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_std]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_std'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        continue
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
                    exec_cols.append((col, num_col))

        def func(df,useful_cols):
            col = df.columns[0]
            num_col = df.columns[1]
            
            df[num_col] = df[num_col].astype('float32')
            
            std = df.groupby(col,sort=False)[num_col].std()
            ss = df[col].map(std)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]], useful_cols) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            del res
            gc.collect()
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
            
            del tmp
            gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []
    
    @timeclass(cls='KeysNumStd')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
         
class KeysCount(Feat):
    '''
    ????????????count???
    '''
    @timeclass(cls='KeysCount')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysCount')
    def transform(self,X):
        df = X.data
        col2type = {}
        col2groupby = {}
        todo_cols = X.session_cols + X.key_cols + X.user_cols

        for col in todo_cols:
            col_count = df[col].value_counts()
            new_col = col+'_KeysCount'
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)

            df[new_col] =  downcast(df[col].map(col_count))
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            col2groupby[new_col] = col

        if len(col2type)>0:
            X.update_data(df,col2type,col2groupby,col2source_cat=col2groupby)

    @timeclass(cls='KeysCount')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)
        
class UserKeyCnt(Feat):
    '''
    ????????????count???????????????*max???????????????+?????????????????????????????????????????????count
    '''
    @timeclass(cls='UserKeyCnt')
    def fit(self,X,y):
        pass

    @timeclass(cls='UserKeyCnt')
    def transform(self,X):
        if not X.user_cols:
            return
        
        user_cols = sorted(X.user_cols)
        key_cols = sorted(X.key_cols)

        col2type = {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for user_col in user_cols:
            for key_col in key_cols:
                new_col = '{}_{}_cnt'.format(user_col, key_col)
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                exec_cols.append((user_col, key_col))
                new_cols.append(new_col)
                col2groupby[new_col] = (user_col,key_col)
                
        df = pd.DataFrame(index=X.data.index)
        for col in user_cols+key_cols:
            df = pd.concat([df,X.data[col]],axis=1)
            
        
        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            ss = cats.map(cnt)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df, col2type,col2groupby)
            
    @timeclass(cls='UserKeyCnt')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)
    
class SessionKeyCnt(Feat):
    '''
    ????????????????????????userkeycnt?????????????????????????????????usercol?????????sessioncol
    '''
    @timeclass(cls='SessionKeyCnt')
    def fit(self,X,y):
        pass

    @timeclass(cls='SessionKeyCnt')
    def transform(self,X):
        if not X.session_cols:
            return

        session_cols = sorted(X.session_cols)
        key_cols = sorted(X.key_cols)

        col2type = {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for session_col in session_cols:
            for key_col in key_cols:
                new_col = '{}_{}_cnt'.format(session_col, key_col)
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                exec_cols.append((session_col, key_col))
                new_cols.append(new_col)
                col2groupby[new_col] = (session_col,key_col)
            
        df = pd.DataFrame(index=X.data.index)
        for col in session_cols+key_cols:
            df = pd.concat([df,X.data[col]],axis=1)
            

        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            ss = cats.map(cnt)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df, col2type,col2groupby)

    @timeclass(cls='SessionKeyCnt')
    def fit_transform(self, X,y):
        self.transform(X)

class UserSessionNuniqueDIY(Feat):
    '''
    ???user groupby session???????????????nunique?????????????????????
    '''
    @timeclass(cls='UserSessionNuniqueDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='UserSessionNuniqueDIY')
    def transform(self,X):

        user_cols = X.user_cols
        session_cols = X.session_cols

        col2type =  {}
        col2groupby = {}

        exec_cols = []
        new_cols = []


        for col1 in user_cols:
            for col2 in session_cols:
                   exec_cols.append((col1, col2))
                   new_col = '{}_{}_Nunique'.format(col1, col2)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col1
                   new_cols.append(new_col)

        df = pd.DataFrame(index=X.data.index)
        for col in user_cols+session_cols:
            df = pd.concat([df,X.data[col]],axis=1)
        
        def func(df):
            col1 = df.columns[0]
            col2 = df.columns[1]
            group = df.groupby(col1)[col2]
            ss = group.nunique()
            new_col = '{}_{}_Nunique'.format(col1, col2)
            ss = downcast(ss)
            ss = df[col1].map(ss)
            ss.name = new_col
            return ss
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for new_col in new_cols:
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
                
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='UserSessionNuniqueDIY')
    def fit_transform(self,X,y):
        self.transform(X)

class UserSessionCntDivNuniqueDIY(Feat):
    '''
    ?????????user groupby session??????????????????count/nunique?????????????????????
    '''
    @timeclass(cls='UserSessionCntDivNuniqueDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='UserSessionCntDivNuniqueDIY')
    def transform(self,X):

        user_cols = X.user_cols
        session_cols = X.session_cols

        col2type =  {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for col1 in user_cols:
            for col2 in session_cols:
                   exec_cols.append((col1, col2))
                   new_col = '{}_{}_CntDivNunique'.format(col1, col2)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col1
                   new_cols.append(new_col)

        df = pd.DataFrame(index=X.data.index)
        for col in user_cols+session_cols:
            df = pd.concat([df,X.data[col]],axis=1)

        def func(df):
            col1 = df.columns[0]
            col2 = df.columns[1]
            group = df.groupby(col1)[col2]
            ss = group.count() / group.nunique()
            new_col = '{}_{}_CntDivNunique'.format(col1, col2)
            ss = downcast(ss)
            ss = df[col1].map(ss)
            ss.name = new_col
            return ss
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for new_col in new_cols:
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='UserSessionCntDivNuniqueDIY')
    def fit_transform(self,X,y):
        self.transform(X)

class KeysNumMeanMinus(Feat):
    @timeclass(cls='KeysNumMeanMinus')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysNumMeanMinus')
    def transform(self,X,useful_cols=None):
        df = X.data

        todo_cols = X.user_cols+X.key_cols+X.session_cols
        num_cols = X.combine_num_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_max]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_mean'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)
    
                    new_col = '{}_{}_meanMinusSelf'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
        else:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_max]
                for num_col in cur_num_cols:
                    new_col1 = '{}_{}_mean'.format(col, num_col)
                    new_col1 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col1,CONSTANT.NUMERICAL_TYPE)
                    if new_col1 not in useful_cols:
                        do1 = False
                    else:
                        do1 = True
                        new_cols.append(new_col1)
                        col2type[new_col1] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col1] = col
    
                    new_col = '{}_{}_meanMinusSelf'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        do2 = False
                    else:
                        do2 = True 
                        new_cols.append(new_col)
                        col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col] = col
                        
                    if not (do1 | do2):
                        continue
                    
                    exec_cols.append((col, num_col))

        def func(df,useful_cols):
            col = df.columns[0]
            num_col = df.columns[1]
            
            df_new = pd.DataFrame(index=df.index)
            
            df[num_col] = df[num_col].astype('float32')
            
            means = df.groupby(col,sort=False)[num_col].mean()
            ss = df[col].map(means)
            
            df_new['new'] = downcast(ss)
            df_new['minus'] = downcast(ss-df[num_col])
            
            if useful_cols is None:
                return df_new
            
            new_col1 = '{}_{}_mean'.format(col, num_col)
            new_col1 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col1,CONSTANT.NUMERICAL_TYPE)
            new_col = '{}_{}_meanMinusSelf'.format(col, num_col)
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
            if new_col1 in useful_cols and new_col in useful_cols:
                return df_new
            elif new_col1 not in useful_cols:
                return df_new['new']
            elif new_col not in useful_cols:
                return df_new['minus']
            else:
                return None

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]], useful_cols) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            del res
            gc.collect()
        
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
            
            del tmp
            gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []
    
    @timeclass(cls='KeysNumMeanMinus')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class KeysNumMaxMinMinus(Feat):
    @timeclass(cls='KeysNumMaxMinMinus')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysNumMaxMinMinus')
    def transform(self,X,useful_cols=None):
        df = X.data

        todo_cols = X.user_cols+X.key_cols+X.session_cols
        num_cols = X.combine_num_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_maxmin]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_max'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)
                    
                    new_col = '{}_{}_min'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
                    
                    new_col = '{}_{}_maxMinusSelf'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
                    
                    new_col = '{}_{}_minMinusSelf'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
        else:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_maxmin]
                for num_col in cur_num_cols:
                    new_col1 = '{}_{}_max'.format(col, num_col)
                    new_col1 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col1,CONSTANT.NUMERICAL_TYPE)
                    if new_col1 not in useful_cols:
                        do1 = False
                    else:
                        do1 = True
                        new_cols.append(new_col1)
                        col2type[new_col1] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col1] = col
                        
                    new_col2 = '{}_{}_min'.format(col, num_col)
                    new_col2 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col2,CONSTANT.NUMERICAL_TYPE)
                    if new_col2 not in useful_cols:
                        do2 = False
                    else:
                        do2 = True
                        new_cols.append(new_col2)
                        col2type[new_col2] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col2] = col
                    
                    new_col3 = '{}_{}_maxMinusSelf'.format(col, num_col)
                    new_col3 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col3,CONSTANT.NUMERICAL_TYPE)
                    if new_col3 not in useful_cols:
                        do3 = False
                    else:
                        do3 = True
                        new_cols.append(new_col3)
                        col2type[new_col3] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col3] = col
                        
                    new_col4 = '{}_{}_minMinusSelf'.format(col, num_col)
                    new_col4 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col4,CONSTANT.NUMERICAL_TYPE)
                    if new_col4 not in useful_cols:
                        do4 = False
                    else:
                        do4 = True
                        new_cols.append(new_col4)
                        col2type[new_col4] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col4] = col
                        
                    if not (do1 | do2 | do3 | do4):
                        continue
                        
                    exec_cols.append((col, num_col))

        def func(df,useful_cols):
            col = df.columns[0]
            num_col = df.columns[1]
            
            df_new = pd.DataFrame(index=df.index)
            
            df[num_col] = df[num_col].astype('float32')
            
            maxs = df.groupby(col,sort=False)[num_col].max()
            mins = df.groupby(col,sort=False)[num_col].min()
            ss_max = df[col].map(maxs)
            ss_min = df[col].map(mins)
            
            new_col1 = '{}_{}_max'.format(col, num_col)
            new_col1 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col1,CONSTANT.NUMERICAL_TYPE)
            new_col2 = '{}_{}_min'.format(col, num_col)
            new_col2 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col2,CONSTANT.NUMERICAL_TYPE)
            new_col3 = '{}_{}_maxMinusSelf'.format(col, num_col)
            new_col3 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col3,CONSTANT.NUMERICAL_TYPE)
            new_col4 = '{}_{}_minMinusSelf'.format(col, num_col)
            new_col4 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col4,CONSTANT.NUMERICAL_TYPE)
      
            df_new[new_col1] = downcast(ss_max)
            df_new[new_col2] = downcast(ss_min)
            df_new[new_col3] = downcast(ss_max-df[num_col])
            df_new[new_col4] = downcast(ss_min-df[num_col])
            
            if useful_cols is None:
                return df_new
            
            new_cols = [new_col1, new_col2, new_col3, new_col4]
            exec_cols = []
            for new_col in new_cols:
                if new_col in useful_cols:
                    exec_cols.append(new_col)
            
            if not exec_cols:
                return None
            return df_new[exec_cols]

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]], useful_cols) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            del res
            gc.collect()
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
            
            del tmp
            gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []
    
    @timeclass(cls='KeysNumMaxMinMinus')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class KeysNumStd(Feat):
    @timeclass(cls='KeysNumStd')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysNumStd')
    def transform(self,X,useful_cols=None):
        df = X.data

        todo_cols = X.user_cols+X.key_cols+X.session_cols
        num_cols = X.combine_num_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_std]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_std'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)

        else:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_std]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_std'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        continue
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
                    exec_cols.append((col, num_col))

        def func(df,useful_cols):
            col = df.columns[0]
            num_col = df.columns[1]
            
            df[num_col] = df[num_col].astype('float32')
            
            std = df.groupby(col,sort=False)[num_col].std()
            ss = df[col].map(std)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]], useful_cols) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            del res
            gc.collect()
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
            
            del tmp
            gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []
    
    @timeclass(cls='KeysNumStd')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    