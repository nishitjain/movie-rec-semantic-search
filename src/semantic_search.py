from json import load
from matplotlib.pyplot import hist
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pipeline import Processor

class SemanticSearch():
    def __init__(self, data_load_path, model_load_path, embeddings_save_path, load_embeddings):
        self.processor = Processor(data_load_path, model_load_path, embeddings_save_path, load_embeddings)
        self.processor.get_embeddings()
        
    def get_hits(self, queries, k):
        queries = [query for query in queries if query != ""]
        query_embeddings = self.processor.model.encode(queries, normalize_embeddings=True)
        hits = util.semantic_search(query_embeddings, self.processor.corpus_embeddings, top_k=k)
        return [(query, sorted(hit, key=lambda x: x['score'], reverse=True)) for (query, hit) in zip(queries, hits)]
