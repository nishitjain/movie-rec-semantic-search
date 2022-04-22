from pipeline import Processor
from annoy import AnnoyIndex

class SemanticSearch():
    def __init__(self, load_index, dim, data_load_path, model_load_path, embeddings_save_path, index_save_path, preprocess_data, load_embeddings):
        self.processor = Processor(data_load_path, model_load_path, embeddings_save_path, preprocess_data, load_embeddings)
        self.processor.get_embeddings()
        self.index_save_path = index_save_path
        self.load_index = load_index
        self.index = self.get_index(dim)
        
    def get_hits(self, queries, k):
        queries = [query for query in queries if query != ""]
        query_embeddings = self.processor.model.encode(queries, normalize_embeddings=True)
        hits = []
        for i in range(len(queries)):
            nn_indices = self.index.get_nns_by_vector(query_embeddings[i], k)
            hits.append((queries[i], nn_indices))
        return hits

    def get_index(self, dim):
        index = AnnoyIndex(dim, metric="angular")
        if self.load_index:
            index.load(self.index_save_path)
        else:
            for i in range(self.processor.corpus_embeddings.shape[0]):
                index.add_item(i, self.processor.corpus_embeddings[i])
            index.build(100)
            index.save(self.index_save_path)
        return index

    
