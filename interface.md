# AutoFeatureExtractor

## Module **main**

### Class **AFE**

```python
__init__(
    info=dict
    )
```

#### the process: 
1. define class table
2. use merger.py to merge input table. During the merge process, graph.py was used to do the dfs merge; and merge_feat.py to generate some nessacary merge feature.
3. preprocess the merged table
4. use feat.py to generate new features
5. use feat_select.py to select feature, it use the auto LGB model which lies in autoModel.py.
6. output the result

## Module **preprocess**

### Class **GeneralPreprocessor**

```python
__init__(
)
```

#### Function
+ `transform`: take a two dimention data and transform it into certain data type
+ `fit_transform`: fit data and transform

### Class **BinaryPreprocessor**

```python
__init__(
)
```

#### Function
+ `transform`: take a two dimention data and process binary data
+ `fit_transform`: fit data and transform

### Class **MSCatPreprocessor**

```python
__init__(
)
```

#### Function
+ `transform`: take a two dimention data and process multi-catagory data
+ `fit_transform`: fit data and transform

### Class **NumPreprocessor**

```python
__init__(
)
```

#### Function
+ `transform`: take a two dimention data and process numberic data
+ `fit_transform`: fit data and transform

## Module **feat**

### Class **CatNumStatistic**

```python
__init__(
    config=list
)
```

#### Function
+ `transform`: take a two dimention data and calculate category data and numberic data combined into new feature
+ `fit_transform`: fit data and transform

### Class **KeyNumStd**

```python
__init__(
    config=list
)
```

#### Function
+ `transform`: take a two dimention data and calculate category data and numberic data std feature
+ `fit_transform`: fit data and transform

### Class **KeysCount**

```python
__init__(
    config=list
)
```

#### Function
+ `transform`: take a two dimention data and calculate category data count feature
+ `fit_transform`: fit data and transform

### Class **UserKeyCnt**

```python
__init__(
    config=list
)
```

#### Function
+ `transform`: user column's max value concate with category columns, then count this new column
+ `fit_transform`: fit data and transform

### Class **SessionKeyCnt**

```python
__init__(
    config=list
)
```

#### Function
+ `transform`: same as UserKeyCnt, but use session column instead of user column
+ `fit_transform`: fit data and transform

### Class **UserSessionNuniqueDIY**

```python
__init__(
    config=list
)
```

#### Function
+ `transform`: use user column groupby session column, then calculate nunique (count different value)
+ `fit_transform`: fit data and transform

### Class **UserSessionCntDivNuniqueDIY**

```python
__init__(
    config=list
)
```

#### Function
+ `transform`: use user column groupby session column, then calculate count/nunique (count different value)
+ `fit_transform`: fit data and transform

### Class **KeysNumMeanMinus**

```python
__init__(
    config=list
)
```

#### Function
+ `transform`: use category column groupby numberic column, then calculate the average
+ `fit_transform`: fit data and transform

## Module **feat_Selection**

### Class **LGBFeatureSelection**

```python
__init__(
    X=table;
    y=numpy
    )
```

#### Function
+ `fit`: use LGB model calculate the importance of feature
+ `transform`: drop least important feature
+ `fit_transform`: fit data and transform