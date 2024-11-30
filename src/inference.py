import pandas as pd
from config import missing_maps, value_maps, features
from utils import maps_labels, fill_missing_values, load_model, load_encoders, encode_live_data

class churn_predictor:
    def __init__(self, data_path, encoders_path, model_path):
        
        self.data_path = data_path
        self.encoders_path = encoders_path
        self.model_path = model_path
        self.encoders = load_encoders(self.encoders_path)
        self.model = load_model(self.model_path)

    def predict(self):
        data = pd.read_csv(self.data_path)
        data_processes = data[features]
        data_processes = maps_labels(data_processes, value_maps)
        data_processes = encode_live_data(data_processes, self.encoders)
        data_processes = fill_missing_values(data_processes, missing_maps)
        data['prediction'] = self.model.predict(data_processes)
        return data

if __name__ == "__main__":
    data_path = 'live_data.csv' 
    encoders_path = 'models/encoders.pkl'
    model_path = 'models/gmm.pkl'

    predictor = churn_predictor(data_path, encoders_path, model_path)
    predictions = predictor.predict()
    print(predictions)
