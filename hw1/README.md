# Machine Learning (2023fall) – Assignment I
## 作者
RE6124035 數據所 黃亮臻

## 作業內容
本次作業(參照 ML-Assignment1.pdf) 分為三大部分：
1. **Classification Task**:  
手刻實現 Linear Classifier, K-NN Classifier, Naïve Decision Tree Classifier 以及 Decision Tree with Pruning。
2. **Feature Engineering**:  
使用 Linear Classifiers, Decision Tree 以及 SHAP 計算特徵重要性，並使用挑選出的重要變數嘗試提高模型表現。
3. **Cross-Validation**:  
檢驗模型的穩健性與是否過度擬合。  


## 檔案說明
1. 於 ML-HW1_黃亮臻.ipynb 作答作業問題，手刻模型儲存於 model.py，完整報告書見 ML-HW1_黃亮臻.pdf。  
2. 使用 train.csv，切分 80%/20% 作為訓練集與測試集。  
資料來源：[kaggle: Car Insurance Claim Prediction
](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification?select=train.csv)

## 環境
【Python】3.11.5  
【Pandas】2.0.3  
【NumPy】1.24.3  
【SHAP】0.43.0  
【Matplotlib】3.7.2  
【Scikit-learn】1.3.0  

## 簡介與操作說明
### Classification Task
### 1. Linear Classifier: 
- Parameters:
    - **lr**: 學習率。預設為 0.02。
    - **epoch**: 迭代次數。預設為 100。
- method:
    - **fit**: 訓練模型的函數  
    - **predict**: 對新數據進行預測的函數，可根據其結果另外計算 Accuracy 與 f1 score
    - **get_coefficient**: 返回模型權重
    - **get_loss**: 返回模型在訓練過程中的損失

### 2. Knn Classifier:  
- Parameters:
    - **k**: 鄰居數量。預設為 3。
    - **metrics**: 測量距離方式。可以選擇 'euclidean'（歐式距離）、 'manhattan'（曼哈頓距離）或 'cosine'（餘弦相似度）。預設為 'euclidean'。
- method:
    - **fit**: 訓練模型的函數  
    - **predict**: 對新數據進行預測的函數，可根據其結果另外計算 Accuracy 與 f1 score
 
### 3. Naïve Decision Tree Classifier:  
- Parameters:
    - **measure**: 決策樹分割時所用的準則。可選擇訊息增益（'information gain'）或 gini index（'gini')。預設為 'information gain'。

- method:
    - **fit**: 訓練模型的函數  
    - **predict**: 對新數據進行預測的函數，可根據其結果另外計算 Accuracy 與 f1 score
    - **get_feature_importances**: 獲得特徵重要性的函數


### 4. Decision Tree with Pruning:  
- Parameters:
    - **measure**: 決策樹分割時所用的準則。可選擇訊息增益（'information gain'）或 gini index（'gini')。預設為 'information gain'。
    - **max_depth**: 樹的最大深度。預設為無限
    - **min_samples_split**: 繼續劃分的最小樣本數。預設為2

- method:
    - **fit**: 訓練模型的函數  
    - **predict**: 對新數據進行預測的函數，可根據其結果另外計算 Accuracy 與 f1 score
    - **get_feature_importances**: 獲得特徵重要性的函數



###  Feature engineering
- 印出 Linear Classifier, Naive decision trees, Decision Tree with Pruning 各個變數的 feature importances 長條圖。  
可調整 `featureplot(importance, ylabel, title, k)` 中的 k 決定欲印出幾個重要變數。

- SHAP  
因手刻模型與 SHAP 的接口不兼容，因此使用 `shap.Explainer(clf.predict, X_train)` 方式，計算較耗時。

- Derive new features  
使用以上多種模型與 SHAP 挑出的共同重要特徵作為新資料的 feature。


### Cross-Validation
1. KNN 與 Decision Tree 可透過 `cross_validation()` 計算 k fold 下 Validation set 與 Test set 的表現，以便後續畫圖觀察。

2. Linear Classifier 可透過 `cross_validation_with_loss()` 計算 k fold 下 Validation set 與 Test set 的表現，也可另外搭配使用`cal_loss_cv`與`plot_losses()` 函數畫出不同 fold 的 epoch-loss 關係圖。
