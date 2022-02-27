from lib2to3.pytree import Base
from urllib.request import Request
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
from semantic_search import SemanticSearch

app = FastAPI()

DATA_LOAD_PATH = "../data/movies_metadata.csv"
MODEL_LOAD_PATH = "../model/all-distilroberta-v1"
EMBEDDING_SAVE_PATH = "../data/corpus_embeddings.npz"
LOAD_EMBEDDINGS = True

search_obj = SemanticSearch(DATA_LOAD_PATH, MODEL_LOAD_PATH, EMBEDDING_SAVE_PATH, LOAD_EMBEDDINGS)

class Movie(BaseModel):
    title: str
    release_date: str
    genre: List[str]
    rating: float

class SingleResult(BaseModel):
    rank: int
    movie: Movie

class QueryResult(BaseModel):
    query: str
    movies: List[SingleResult]

class SearchRequest(BaseModel):
    search_texts: List[str]

class SearchResponse(BaseModel):
    results: List[QueryResult] = []

class SearchError(BaseModel):
    error_code: int
    status: str
    description: str

@app.post("/search")
async def get_matches(request: SearchRequest):
    if len(request.search_texts) == 0:
        error_dict = {"error_code": 400, "status": "INVALID_REQUEST", "description": "Missing Data"}
        return SearchError(**error_dict)
    else:
        results = search_obj.get_hits(queries=request.search_texts, k=10)
        response = []
        for result in results:
            movies = []
            query = result[0]
            for hit in result[1]:
                title = search_obj.processor.data.iloc[hit['corpus_id']]['original_title']
                release_date = str(search_obj.processor.data.iloc[hit['corpus_id']]['release_date'])
                genre = search_obj.processor.data.iloc[hit['corpus_id']]['genre']
                rating = 3.5 # search_obj.processor.data.iloc[hit['corpus_id']]['avg_user_rating']
                movie_dict = {"title": title, "release_date": release_date, "genre": genre, "rating": rating}
                movie = Movie(**movie_dict)
                rank = result[1].index(hit) + 1
                single_result_dict = {"rank": rank, "movie": movie}
                movies.append(SingleResult(**single_result_dict))
            query_result_dict = {"query": query, "movies": movies}
            response.append(QueryResult(**query_result_dict))
        response_dict = {"results": response}
        return SearchResponse(**response_dict)
