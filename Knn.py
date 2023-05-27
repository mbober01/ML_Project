from utils import Utils, np, pd
class Knn:
    def __init__(self, k=3, m=1):
        self.k = k
        self.m = m
        self.y = None
        self.X_train = None
        self.y_train = None
        self.cols = None
        self.train = None

    def fit(self, train, y=None):
        if y:
            self.y = y
        else:
            self.y = train.columns[-1]
        self.X_train = train.drop(self.y,axis=1)
        self.y_train = train[self.y]
        self.train = train

    def predict(self, test):
        X_test = test.drop(self.y, axis=1)
        predictions = []
        for record in X_test.to_numpy():
            dist = []
            for train_record in self.X_train.to_numpy():
                dist.append(Utils.distance(record,train_record,self.m))
            args = np.argsort(dist)
            result = np.mean(self.y_train.iloc[args[:5]])
            predictions.append(result)
        return predictions

    def mae(self, test, pred):
        test = test[self.y]
        errors = []
        for i,val in enumerate(test):
            error = np.abs(test.iloc[i] - pred[i])
            errors.append(error)

        return np.mean(errors)

    def rmse(self,test,pred):
        test= test[self.y]
        errors = []
        for i,val in enumerate(test):
            error = pow(test.iloc[i] - pred[i], 2)
            errors.append(error)

        return np.sqrt(np.mean(errors))