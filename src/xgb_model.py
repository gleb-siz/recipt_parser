import os
import cv2
import json
import re
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("XGB Trainer")

class XGB_model:


    def __init__(self, name, models_path, xgb_paramters):
        self.name = name
        self.models_path = models_path
        self.xgb_parameters = xgb_paramters
        self.model = None
        self.check_dir(models_path)
        self.check_dir(f"{models_path}/{name}")
        self.check_dir(f"{self.models_path}/{self.name}/archive")


    def check_dir(self, path):
        try:
            os.listdir(path)
        except FileNotFoundError as e:
            logger.error(f"Folder {path} not found creating folder...")
            os.mkdir(path)


    def train_model(self, X, y, X_test, y_test, xgb_param_override):
        
        id = datetime.now().strftime("%Y%m%d%H%M")
        # self.check_dir(f"{self.models_path}/{self.name}/{id}")
        
        
        le = LabelEncoder()
        y_train = le.fit_transform(y)
        y_test = le.fit_transform(y_test)
        classes = np.unique(y_train)
        weights = compute_class_weight( 'balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        # Assign weight to each sample
        sample_weights = np.array([class_weight_dict[label] for label in y_train])

        self.model = xgb.XGBClassifier(
            # objective="multi:softmax",
            objective="binary:logistic",
            n_estimators=500,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            use_label_encoder=False,
            early_stopping_rounds=50,
            **xgb_param_override
        )
        _, X_val, _, y_val = train_test_split(
            X_test, y_test, test_size=0.2, stratify=y_test, random_state=42
        )
        self.model.fit(X, 
                y_train, 
                eval_set=[(X_val, y_val),
                            ],
                sample_weight=sample_weights,
                verbose=5
                )
        
        self.log_model(id=id)
        log = self.evaluate_model(X_test, y_test)
        self.log_evaluation(id=id, log=log)
        

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        report= classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred, 
                                # multi_class='ovr'
                                )  # or 'ovo'
        return {**report, 'ROC_AUC': roc_auc}


    def log_model(self, id):
        best = f"{self.models_path}/{self.name}/best.ubj"
        archive = f"{self.models_path}/{self.name}/archive/{id}/best.ubj"
        self.check_dir(f"{self.models_path}/{self.name}/archive/{id}")
        self.model.save_model(best)
        self.model.save_model(archive)


    def log_evaluation(self, id, log):
        archive = f"{self.models_path}/{self.name}/archive/{id}/log.json"
        with open(archive, 'w') as f:
            f.write(json.dumps(log))


    def load_model(self, id='best'):
        self.model = xgb.XGBClassifier()
        self.model = self.model.load_model(f"{self.models_path}/{self.name}/{id}")

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred




    
