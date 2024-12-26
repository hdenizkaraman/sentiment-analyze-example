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
    
    def __detect_encoding(self) -> None:
        """
        Detects encoding of the dataset file.
        We must know that to read file properly.
        """
        try:
            with open(self.datasetPath, 'rb') as f:
                self.encoding = chardet.detect(f.read())['encoding']
        except FileNotFoundError:
            print("Dataset not found")
            self.encoding = None
        
    def __load_dataset(self) -> None:
        """
        After detecting encoding, loads dataset from the given path.
        """
        self.__detect_encoding()
        if self.encoding:
            try:
                self.dataset = pd.read_csv(self.datasetPath, encoding=self.encoding)
            except Exception as e:
                print(f"Error reading dataset: {e}")
                self.dataset = None
    
    @__check_dataset_loaded
    def __split_dataset_train_test(self) -> tuple:
        """
        We split dataset into two parts: train and test.
        """
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
    

dataset = Dataset('dataset.csv', 80)
splittedData = dataset.get_reviews_and_labels()

class SentimentAnalyzer:
    def __init__(self, dataset:tuple)->None:
        self.trainReviews, self.trainLabels, self.testReviews, self.testLabels = dataset
        self.maxTokens = 10000
        self.outputDim = 50
        self.countedWords = None
        self.outputLength = None
        self.tokenizer = None
        self.model = None
        
        self.numberizedTrainReviews = None
        self.numberizedTestReviews = None
        
    def __count_words(self):
        """
        Counts words in every single review.
        """
        self.countedWords = np.vectorize(lambda x: len(x.split()))(self.trainReviews)
        
    def __calculate_output_length(self):
        """
        Cuts or pads reviews to the same length.
        In this case, we calculate the mean length of the reviews and add one standard deviation.
        """
        mean_length = np.mean(self.countedWords)
        std_length = np.std(self.countedWords)
        self.outputLength = int(mean_length + std_length)  
    
    def tokenize(self):
        self.__count_words()
        self.__calculate_output_length()
        self.tokenizer = tf.keras.layers.TextVectorization(max_tokens=self.maxTokens, 
                                              output_sequence_length=self.outputLength)
        self.tokenizer.adapt(self.trainReviews)
        
        self.numberizedTrainReviews = self.tokenizer(self.trainReviews).numpy()
        
    def most_common_words(self):
        """
        Prints 10 most common words in the dataset.
        And returns the vocabulary.
        """
        vocabulary = np.array(self.tokenizer.get_vocabulary())
        print(vocabulary[:10])
        return vocabulary
    
    def build_layers(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.InputLayer(shape=(self.outputLength,),))
        self.model.add(tf.keras.layers.Embedding(input_dim=self.maxTokens,
                                            output_dim=self.outputDim,
                                            name='embedding_layer'
                                            ))

        
        # If the next layer is not GRU, return_sequences must be False.
        self.model.add(tf.keras.layers.GRU(units=16, return_sequences=True))
        self.model.add(tf.keras.layers.GRU(units=8, return_sequences=True))
        self.model.add(tf.keras.layers.GRU(units=4))

        # Final layer, incoming values zip between 0 and 1 with sigmoid function.
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
    
        tensorDatasetTrain = tf.data.Dataset.from_tensor_slices((self.numberizedTrainReviews, self.trainLabels)).batch(256)
        self.model.fit(tensorDatasetTrain, epochs=10)

    def performance_info(self):
        """
        Prints the performance of the model.
        Based on test dataset.
        """
        self.numberizedTestReviews = self.tokenizer(self.testReviews).numpy()
        tensorDatasetTest = tf.data.Dataset.from_tensor_slices((self.numberizedTestReviews, self.testLabels)).batch(256)
        loss, accuracy = self.model.evaluate(tensorDatasetTest)
        print(f"Loss: {loss}\nAccuracy: {accuracy}")
    
analyzer = SentimentAnalyzer(splittedData)
analyzer.tokenize()
analyzer.build_layers()
analyzer.performance_info()
