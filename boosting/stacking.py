from sklearn.model_selection import KFold
import numpy as np
import copy


class Stacking:

    def __init__(self, seed=0):
        self.final_base_models = []
        self.seed = seed

    def fit(self, X, y, base_models, meta_model, K):
        number_of_samples = X.shape[0]
        number_of_base_models = len(base_models)
        Z = np.zeros(number_of_samples, number_of_base_models)
        
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
    
        for k in kf:
            train_index, validation_index = self.__split(k)
            
            for b_model in base_models:
                model_b_k = copy.deepcopy(b_model)
                model_b_k.fit(X[train_index], y[train_index])
                
                Z[validation_index, b_model] = model_b_k.predict(X[validation_index])
                
            meta_model.fit(Z, y)
            
            for b_model in base_models:
                model_b_full = copy.deepcopy(b_model)
                model_b_full.fit(X, y)
                self.final_base_models.append(model_b_full)
            return self.final_base_models, meta_model
    
    def __split(self, k):
        pass
    
    def predict(self, final_base_models, meta_model, X):
        Z_new = [model_b_full.predict(X) for model_b_full in final_base_models]
        return meta_model.predict(Z_new)


if __name__ == "__main__":
    pass
