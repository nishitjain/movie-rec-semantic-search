from fastapi import FastAPI
from semantic_search import SemanticSearch
from schema import *
from constants import *

app = FastAPI()

search_obj = SemanticSearch(LOAD_INDEX, DIM, DATA_LOAD_PATH, MODEL_LOAD_PATH, EMBEDDING_SAVE_PATH, INDEX_SAVE_PATH, PREPROCESS_DATA, LOAD_EMBEDDINGS)

@app.post("/search")
async def get_matches(request: SearchRequest):
    if len(request.search_texts) == 0:
        error_dict = {"error_code": 400, "status": "INVALID_REQUEST", "description": "Missing Data"}
        return SearchError(**error_dict)
    else:
        results = search_obj.get_hits(queries=request.search_texts, k=K)
        response = []
        for result in results:
            movies = []
            query = result[0]
            for hit in result[1]:
                title = search_obj.processor.data.iloc[hit]['original_title']
                overview = search_obj.processor.data.iloc[hit]['overview']
                tagline = search_obj.processor.data.iloc[hit]['tagline']
                release_date = str(search_obj.processor.data.iloc[hit]['release_date'])
                genre = search_obj.processor.data.iloc[hit]['genre']
                vote_average = search_obj.processor.data.iloc[hit]['vote_average']
                vote_count = search_obj.processor.data.iloc[hit]['vote_count']
                movie_dict = {"title": title, "overview": overview, "tagline": tagline, "release_date": release_date, "genre": genre, "vote_average": vote_average, "vote_count": vote_count}
                movie = Movie(**movie_dict)
                rank = result[1].index(hit) + 1
                single_result_dict = {"rank": rank, "movie": movie}
                movies.append(SingleResult(**single_result_dict))
            query_result_dict = {"query": query, "movies": movies}
            response.append(QueryResult(**query_result_dict))
        response_dict = {"results": response}
        return SearchResponse(**response_dict)
