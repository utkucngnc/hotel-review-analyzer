from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import pickle

class BasicClassifier:
    def __init__(self, df, heads) -> None:
        self.df = df
        self.models = {
                        'nb': Pipeline([('vectorize', CountVectorizer(ngram_range=(1, 2))),
                                        ('tfidf', TfidfTransformer()),
                                        ('clf', MultinomialNB()),
                                    ]),
                        'sgd': Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                                        ('tfidf', TfidfTransformer()),
                                        ('clf', SGDClassifier()),
                                        ]),
                        'logreg': Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                                            ('tfidf', TfidfTransformer()),
                                            ('clf', LogisticRegression(max_iter=500)),
                                        ]),
                    }
        self.model_names =  {
                            'nb': 'Naive Bayes',
                            'sgd': 'Stochastic Gradient Descent',
                            'logreg': 'Logistic Regression',
                            }
        self.heads = heads
    
    def eval_best(self, save_path = None):
        X_test, X_train, y_test, y_train = train_test_split(
                                                    self.df[self.heads[0]], self.df[self.heads[1]], 
                                                    test_size=0.2, stratify=self.df[self.heads[1]], 
                                                    random_state = 44
                                                    )
        max_score = 0.0
        best_model = None
        conf_mat = None

        for name, model in self.models.items():
            print(f"Evaluating {self.model_names[name]}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if accuracy_score(y_test, y_pred) > max_score:
                max_score = accuracy_score(y_test, y_pred)
                best_model = model
                conf_mat = confusion_matrix(y_test, y_pred)
        
        if save_path:
            with open(save_path,'wb') as f:
                pickle.dump(best_model,f)

        return max_score, best_model, conf_mat
    
    def load_model(self, load_path):
        with open(load_path,'rb') as f:
                model = pickle.load(f)
        return model