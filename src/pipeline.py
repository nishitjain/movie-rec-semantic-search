from json import load
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

class Processor():
    def __init__(self, data_load_path, model_load_path, embeddings_save_path, preprocess_data, load_embeddings):
        self.preprocess_data = preprocess_data
        self.data_load_path = data_load_path
        self.data = pd.read_csv(self.data_load_path)
        self.save_path = embeddings_save_path
        self.embeddings_save_path = embeddings_save_path
        self.load_embeddings = load_embeddings
        self.model = SentenceTransformer(model_load_path)

    def parse_dict(self, d, key):
        try:
            name = eval(d)[key]
            return name
        except:
            return ""

    def parse_lst_dict(self, lst, key):
        names = []
        try:
            for d in eval(lst):
                names.append(d[key])
        except:
            pass
        return names

    def preprocess(self):
        if self.preprocess_data:
            self.data['genre'] = self.data['genres'].apply(lambda x: self.parse_lst_dict(x, 'name'))
            self.data['languages'] = self.data['spoken_languages'].apply(lambda x: self.parse_lst_dict(x, 'name'))
            self.data.drop(['belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'original_language', 'production_companies',\
                'production_countries', 'title', 'video', 'poster_path', 'spoken_languages'], axis=1, inplace=True)
            self.data.dropna(subset='overview', inplace=True)
            self.data.to_csv(self.data_load_path.split(".csv")[0]+"_preprocessed.csv", index=False)
        
    def get_embeddings(self):
        self.preprocess()
        if self.load_embeddings:
            self.corpus_embeddings = np.load(self.embeddings_save_path)['arr_0']
        else:
            self.corpus_embeddings = self.model.encode(self.data['overview'].values.tolist(), normalize_embeddings=True, batch_size=128)
            np.savez_compressed(self.embeddings_save_path, self.corpus_embeddings)
