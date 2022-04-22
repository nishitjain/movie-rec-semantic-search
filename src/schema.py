from pydantic import BaseModel
from typing import List

class Movie(BaseModel):
    title: str
    overview: str
    tagline: str
    release_date: str
    genre: List[str]
    vote_average: float
    vote_count: int

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