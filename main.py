
#! Required libraries are imported.
import numpy as np
import pandas as pd
import tensorflow as tf
import chardet
from functools import wraps

#! Dataset class is created to load the dataset.
class Dataset:
    def __init__(self, datasetPath:str, trainDataPercentage:int)->None:
        """
        Initializes the dataset.
        
        Args:
            datasetPath (str): Path of the dataset file.
            trainDataPercentage (int): Percentage of the train data.
        """
        self.datasetPath = datasetPath
        self.trainDataPercentage = trainDataPercentage
        self.dataset = None
        self.encoding = None
        self.__load_dataset()
    
    @staticmethod
    def __check_dataset_loaded(func):
        """
        Decorator to check if the dataset is loaded.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.dataset is None:
                print("Problem: Dataset is not loaded")
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
        trainDataPercentage (//100) is used to determine the percentage of the train data.
        
        Returns:
            tuple: train and test datasets
        """
        cutoff:float = len(self.dataset)*self.trainDataPercentage//100
        train = self.dataset[:int(cutoff)]
        test = self.dataset[int(cutoff):]
        return train, test
    
    @__check_dataset_loaded
    def get_reviews_and_labels(self) -> tuple:
        """
        Gets reviews and labels from the dataset.
        
        Returns:
            tuple: trainReviews, trainLabels, testReviews, testLabels
        """
        def review_label_split(review, label) -> tuple: return review.astype(str).values.tolist(), label.apply(lambda x: 1 if x == "Olumlu" else 0).values.tolist()
        train, test = self.__split_dataset_train_test()
        
        trainReviews, trainLabels = review_label_split(train['Görüş'], train['Durum'])
        testReviews, testLabels = review_label_split(test['Görüş'], test['Durum'])        

        return trainReviews, trainLabels, testReviews, testLabels
    



#! SentimentAnalyzer class is created to analyze the sentiment of the reviews.
class SentimentAnalyzer:
    def __init__(self, dataset:tuple)->None:
        """
        Initializes the SentimentAnalyzer.
        
        Args:
            dataset (tuple): Train and test reviews and labels.
        """
        self.trainReviews, self.trainLabels, self.testReviews, self.testLabels = dataset
        self.maxTokens = 10000
        self.outputDim = 50
        self.countedWords = None
        self.outputLength = None
        self.tokenizer = None
        self.model = None
        
        self.numberizedTrainReviews = None
        self.numberizedTestReviews = None
        
    @staticmethod
    def __catch_none_type_error(func):
        """
        Decorator to catch AttributeError or TypeError. It is used to check if the methods are executed in order.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try: return func(self, *args, **kwargs)
            except AttributeError or TypeError as e:
                print("You must execute tokenize method, build_layers method and then other methods.", e)
                return None
        return wrapper  
        
    def __count_words(self):
        """
        Counts words in every single review.
        """
        self.countedWords = np.vectorize(lambda x: len(x.split()))(self.trainReviews)
        
    def __calculate_output_length(self):
        """
        Cutting or padding reviews to the same length.
        In this case, we calculate the mean length of the reviews and add a standard deviation.
        """
        mean_length = np.mean(self.countedWords)
        std_length = np.std(self.countedWords)
        self.outputLength = int(mean_length + std_length)  
    
    @__catch_none_type_error
    def tokenize(self):
        """
        Counts the words, calculates the output length and tokenizes the reviews.
        Uses TextVectorization layer to do that.
        self.numberizedTrainReviews is created and that is equal to the tokenized train reviews in numpy format.
        """
        self.__count_words()
        self.__calculate_output_length()
        self.tokenizer = tf.keras.layers.TextVectorization(max_tokens=self.maxTokens, 
                                              output_sequence_length=self.outputLength)
        self.tokenizer.adapt(self.trainReviews)
        self.numberizedTrainReviews = self.tokenizer(self.trainReviews).numpy()
        self.numberizedTestReviews = self.tokenizer(self.testReviews).numpy()
        
    @__catch_none_type_error
    def most_common_words(self):
        """
        Prints 10 most common words in the dataset.

        Returns:
            numpy.ndarray: Most common words in the dataset.
        """
        vocabulary = np.array(self.tokenizer.get_vocabulary())
        print(vocabulary[:10])
        return vocabulary
    
    @__catch_none_type_error
    def build_layers(self, reinstall:bool=False)->None:
        """
        Builds the layers of the model.
        self.model is created.
        If reinstall is not True, the model is loaded from the saved model.
        Uses GRU layers.
        Sigmoid function is used in the final layer.
        Adam optimizer is used.
        tensorDatasetTrain is a tensor dataset that is created from the numberized train reviews and train labels.
        Training is done with 10 epochs.
        """
        if not reinstall:
            try:
                self.model = tf.keras.models.load_model('model.keras')
                return
            except Exception as e:
                print(f"Error loading model: {e}")
                    
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

    @__catch_none_type_error
    def performance_info(self):
        """
        Prints the performance of the model.
        Based on test dataset.
        """
        tensorDatasetTest = tf.data.Dataset.from_tensor_slices((self.numberizedTestReviews, self.testLabels)).batch(256)
        loss, accuracy = self.model.evaluate(tensorDatasetTest)
        print(f"Loss: {loss}\nAccuracy: {accuracy}")
        
    @__catch_none_type_error
    def predict_test_review(self)->None:
        """
        Predicts the sentiment of the test reviews.
        """
        prediction = self.model.predict(self.numberizedTestReviews[0:1000]).T[0]
        roundPrediction = np.array([1 if i > 0.5 else 0 for i in prediction])
    
    # @__catch_none_type_error
    def predict_review(self, review:str)->str:
        """
        Predicts the sentiment of the given review.
        
        Args:
            review (str): Review to predict.
        
        Returns:
            str: Prediction of the review.
        """
        numberizedReview = self.tokenizer([review]).numpy()
        prediction = self.model.predict(numberizedReview).T[0]
        print(f'Prediction: {"Positive" if prediction > 0.5 else "Negative"}')
        return prediction
    
    @__catch_none_type_error
    def save(self)->None:
        """
        Saves the model.
        """
        self.model.save('model.keras')
    

    