import numpy as np
import pandas as pd
import tensorflow as tf
import chardet
from functools import wraps

class Dataset:
    def __init__(self, datasetPath:str, trainDataPercentage:int)->None:
        self.datasetPath = datasetPath
        self.trainDataPercentage = trainDataPercentage
        self.dataset = None
        self.encoding = None
        self.__load_dataset()
    
    @staticmethod
    def __check_dataset_loaded(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.dataset is None:
                print("Dataset is not loaded")
                return None
            return func(self, *args, **kwargs)
        return wrapper    
    
    def __detect_encoding(self) -> str:
        try:
            with open(self.datasetPath, 'rb') as f:
                self.encoding = chardet.detect(f.read())['encoding']
        except FileNotFoundError:
            print("Dataset not found")
            self.encoding = None
        
    def __load_dataset(self) -> None:
        self.__detect_encoding()
        if self.encoding:
            try:
                self.dataset = pd.read_csv(self.datasetPath, encoding=self.encoding)
            except Exception as e:
                print(f"Error reading dataset: {e}")
                self.dataset = None
    
    @__check_dataset_loaded
    def __split_dataset_train_test(self) -> tuple:
        cutoff:float = len(self.dataset)*self.trainDataPercentage//100
        train = self.dataset[:int(cutoff)]
        test = self.dataset[int(cutoff):]
        return train, test
    
    @__check_dataset_loaded
    def get_reviews_and_labels(self) -> tuple:
        def review_label_split(review, label) -> tuple: return review.astype(str).values.tolist(), label.apply(lambda x: 1 if x == "Olumlu" else 0).values.tolist()
        train, test = self.__split_dataset_train_test()
        
        trainReviews, trainLabels = review_label_split(train['Görüş'], train['Durum'])
        testReviews, testLabels = review_label_split(test['Görüş'], test['Durum'])        

        return trainReviews, trainLabels, testReviews, testLabels
    

dataset = Dataset('dataset.csv')
trainReviews, trainLabels, testReviews, testLabels = dataset.get_reviews_and_labels()


class SentimentAnalyzer:
    def __init__()->None:
        pass