
class Config:
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 0.01
        self.num_epochs = 3
        self.split = 0.8
        self.pathToData = "/home/adanilishin/rollback/mnist_/mnist_dataset"
        self.pathToLogs = "/home/adanilishin/rollback/logs/test"
        self.detect = True
        self.detectionRate = False
        self.rollback = False
