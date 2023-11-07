#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np


# In[10]:


class LinearClassifier(object):
    '''
    Parameters
    ----------
    lr: learning rate (default = 0.02)
    epoch: epoch (default = 100)

    '''
    def __init__(self, lr = 0.02, epoch = 100):
        self.lr = lr
        self.epoch = epoch
        self.loss = None
        
    # weighted sum
    def Sumfun(self, x):
        return np.dot(x, self.W) + self.b #1*m m*1

    # activation function
    def Actfun(self, z):
        if z>=0:
            return 1
        else:
            return 0
        
    # fit
    def fit(self, X, y, X_val=None, y_val=None):
        '''
        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples, )
            
        Return: 
            None
        '''
        self.loss = np.zeros(self.epoch) # 每個特徵初始化 
        self.W = np.random.rand(X.shape[1]).reshape((X.shape[1],1))
        self.b = np.random.rand(1).reshape((1,1))
        self.validation_loss = np.zeros(self.epoch) if X_val is not None else None

        for i in range(self.epoch):
            epoch_loss = 0
            for xi, yi in zip(X, y):
                z = self.Sumfun(xi)
                y_hat = self.Actfun(z)

                if yi != y_hat:
                    self.W = self.W + self.lr * (yi-y_hat) * xi.reshape((X.shape[1], 1))  # 講義: W = W + lr * y *x.T 
                    self.b = self.b + self.lr * (yi-y_hat)  # 講義: B = B + y
                    epoch_loss += 1   # Increment loss for each misclassification
                    
            self.loss[i] = epoch_loss/len(X)
            
            # 如果有提供 validation, 則提供 validation 的損失
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = np.mean(val_predictions != y_val)  
                self.validation_loss[i] = val_loss
            
        
        return None

    
    # predict
    def predict(self, X_test):
        '''
        Args:
            X_test: array-like of shape (n_samples, n_features)
            
        Return: 
            prediction: array-like of shape (n_samples, )
        '''
        prediction = []
        for i in range(X_test.shape[0]):
            z = self.Sumfun(X_test[i, :])
            y_hat = self.Actfun(z)
            prediction.append(y_hat)
        
        return np.array(prediction)
    
    def get_coefficient(self):
        return self.W.reshape((1, -1))[0].tolist()
                             
    def get_loss(self):
        return self.loss                    


# In[15]:


class KnnClassifier(object):
    '''
    Parameters
    ----------
    k: number of neighbor
    metrics= {'euclidean', 'manhattan', 'cosine'}
    
    '''
    
    def __init__(self, k = 3, metrics ='euclidean'):  
        self.k = k
        self.metrics = metrics
        
    
    def fit(self, X, y):
        '''
        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples, )

        Return: 
            None
        '''
        self.X_train = X
        self.y_train = y

    def distance(self, xi):
        
        # L2: 歐式距離（目的是比大小，就不再另外開根號）
        if self.metrics == 'euclidean':
            dist = np.sum((self.X_train - xi)**2, 1) 

        # L1: 曼哈頓距離
        elif self.metrics == 'manhattan':
            dist = np.sum(abs((self.X_train - xi)), 1) 

        # 餘弦
        elif self.metrics == 'cosine':
            similarity = np.sum(self.X_train * xi, 1) / (np.sqrt(np.sum((self.X_train*self.X_train), 1))*np.sqrt(np.sum(xi*xi)))
            # similarity = np.sum(self.X_train * xi, 1) / (np.sum((self.X_train*self.X_train), 1)*np.sum((xi*xi)))
            dist = 1 - similarity
        
        else:
            raise ValueError("Unsupported metric.")
        
        return dist
                
                
    def predict(self, X_test):
        prediction = []
        for xi in X_test:
            dist = self.distance(xi)
            index_k_y = np.argsort(dist)[:self.k] # 最近k個 分別的位置 (負號算最小)
            k_y = self.y_train[index_k_y] # 最近k個 分別的組別
            y_hat = np.bincount(k_y).argmax() # 哪個組別最多次 （if 平手 自動分給數字最小的組別）
            prediction.append(y_hat)
        return prediction
            
            
    def score(self, X, y):
        # 使用predict方法得到預測結果
        predictions = self.predict(X)
        return np.mean(predictions == y)


# In[3]:


class NaiveDTClassifier(object):
    '''
    Parameters
    ----------
    measure: 'information gain'(default) or 'gini'
    '''
    
    def __init__(self, measure = 'information gain'):
        self.measure = measure
        self.tree = None # 
        self.feature_importances = None #

    def fit(self, X, y):
        '''
        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples, )
        '''
        n_features = X.shape[1]
        self.feature_importances = np.zeros(n_features) # 每個特徵初始化 feature_importances(numpy.ndarray)
        self.tree = self._build_tree(X, y)

    
    def predict(self, X):
        '''
        Args:
            X: array-like of shape (n_samples, n_features)
        Returns:
            prediction of all testing data: array-like of shape (n_samples, )   
        '''
        # 對所有樣本預測
        predicitons = []
        for sample in X:
            pred = self._predict_sample(self.tree, sample)
            predicitons.append(pred)
        
        return np.array(predicitons)

    
    def _metric(self, y):
        _, counts = np.unique(y, return_counts=True)
        small_amount = 1e-9 # 預防分母為0
        p = counts / counts.sum() + small_amount
        
        if self.measure == 'gini':
            score = 1 - np.sum(p**2)
        elif self.measure == 'information gain':
            score = -np.sum((p + small_amount) * np.log(p + small_amount))
        else:
            raise ValueError("Measure must be 'gini' or 'information gain'") 
            
        return score    
        


    # 最佳分割點
    def _best_split(self, X, y):
        best_score = float("inf")
        best_split = None
        parent_score = self._metric(y)

        # 所有變數跑一次
        for feature_idx in range(X.shape[1]):
            #print("feature_idx: ", feature_idx)
            thresholds, counts = np.unique(X[:, feature_idx], return_counts=True) # 所有特徵當作都 thresholds(不重複) 比大小

            # 設定條件: 如果類別 > 10種，就只取最多的10種進 thresholds_10 (>10可能是連續變數，不想全部跑)
            thresholds_10 = thresholds[np.argsort(-counts)][:min(10, len(counts))]
            # print(thresholds_10)
            
            # 不管連續還是類別 所有 unique 的值都當一次 threshold
            for threshold in thresholds_10:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                y_left, y_right = y[left_mask], y[right_mask]

                # 若分割後是空的 跳過
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                left_score = self._metric(y_left)
                right_score = self._metric(y_right)
                weighted_score = (len(y_left) * left_score + len(y_right) * right_score) / X.shape[0]
                
                score_decrease = parent_score - weighted_score # information gain/gini相較原本減少了多少
                # 希望分割後的兩個子節點的 weighted_score比 parent_score 還小
                # score_decrease 越大越好

                if weighted_score < best_score :
                    best_score  = weighted_score
                    best_split = (feature_idx, threshold, score_decrease) # 每進行一次遞迴就算一個 best_split

        # print(best_split)
        return best_split

    # 建樹
    def _build_tree(self, X, y):
        
        # 遞迴終止條件1: 所有 y 的標籤都一樣
        if len(set(y)) == 1:
            return {'label': y[0]}

        # 遞迴終止條件2: 沒有更多的變數用來分割數據
        split = self._best_split(X, y)
        if split is None:
            return {'label': np.bincount(y).argmax()}

        feature, threshold, score_decrease = split
        self.feature_importances[feature] += score_decrease # 同樣的feature有可能會成為兩次以上的節點 都掉算進 importance
        # print(self.feature_importances)
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # print('進行了一次遞迴')
        left_branch = self._build_tree(X[left_mask], y[left_mask])
        right_branch = self._build_tree(X[right_mask], y[right_mask])
        
        return {
        'feature': feature,
        'threshold': threshold,
        'left': left_branch,
        'right': right_branch}
    
                                                         
    def _predict_sample(self, tree, sample):
                                                         
        # 遞迴終止條件: 當進展到 leaf (有class出現時)                                               
        if 'label' in tree:
            return tree['label']
        
        feature, threshold = tree['feature'], tree['threshold']
        
        # 遞迴建左右樹
        if sample[feature] <= threshold:
            return self._predict_sample(tree['left'], sample)
        else:
            return self._predict_sample(tree['right'], sample)
        
    
    def get_feature_importances(self):
        
        total_decrease = np.sum(self.feature_importances)
        return self.feature_importances / total_decrease  


# In[ ]:


class PrunedDTClassifier(NaiveDTClassifier):
    '''
    Parameters
    ----------
    measure: 'information gain'(default) or 'gini'
    max_depth: The maximum depth of the tree (int), default = float('inf')
    min_samples_split: The minimum number of samples required to split an internal node (int), default=2
    
    '''
        
    def __init__(self, measure='information gain', max_depth=float('inf'), min_samples_split=2):
        super().__init__(measure)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    
    def _build_tree(self, X, y, depth=0):
        
        # 遞迴終止條件1: Pruning 達到最大深度 (max_depth)
        if self.max_depth is not None and depth >= self.max_depth:
            return {'label': np.bincount(y).argmax()}

        # 遞迴終止條件2: Pruning 節點必須擁有的最小樣本數 (min_samples_split)
        if len(y) < self.min_samples_split:
            return {'label': np.bincount(y).argmax()}

        # 遞迴終止條件3: 所有 y 的標籤都一樣
        if len(set(y)) == 1:
            return {'label': y[0]}
        
        # 遞迴終止條件4: 沒有更多的變數用來分割數據
        split = self._best_split(X, y)
        if split is None:
            return {'label': np.bincount(y).argmax()}

        feature, threshold, score_decrease = split
        self.feature_importances[feature] += score_decrease

        # 畫分數據
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # 遞迴
        left_branch = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_branch = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_branch,
            'right': right_branch
        }

