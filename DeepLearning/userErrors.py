class NotFittedError(Exception):
    """ Exception raised for errors in the using other estimaters without fitting the model
    
    Attributes:
        message -- explanation of the error
    """
    
    def __init__(self, message = "This DeepNeuralNet instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."):
        self.message = message
        super().__init__(self.message)    