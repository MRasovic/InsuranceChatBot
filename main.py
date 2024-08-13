from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


app = FastAPI()

df = pd.read_csv(r"C:\Users\Korisnik\NLP\ChatBot\products_weights_vocabs\insurance.csv")
insurance_corpus = df.set_index('input')['output'].to_dict()

class QueryRequest(BaseModel):
    question: str
    top_n: int

    @staticmethod
    def return_n_similar_qs(df, question, n):
        pipeline = Pipeline([
            ('bow', CountVectorizer(stop_words='english', lowercase=True)),
            ('tfidf', TfidfTransformer()),
        ])

        piped_matrix = pipeline.fit_transform(df.input)  # Transforming the input vectors
        query_vector = pipeline.transform([question])  # Transforming the query

        sim_matrix = cosine_similarity(query_vector, piped_matrix).flatten()

        top_vector = np.argsort(sim_matrix)[-n:][
                     ::-1]  # gets back indexes of the best n matches. ie cosine similarity descending ([::-1])
        top_matches = [(df.iloc[idx].input, sim_matrix[idx]) for idx in top_vector]

        return top_matches


@app.post("/query", response_model=List[Tuple[str, float]])
async def query(request: QueryRequest):
    if request.top_n <= 0:
        raise HTTPException(status_code=400, detail="top_n must be a positive integer")

    top_matches = QueryRequest.return_n_similar_qs(df, request.question, request.top_n)

    result = [(insurance_corpus.get(question, "No answer found"), score) for question, score in top_matches]

    return result

# Troubleshooting url
# https://www.youtube.com/watch?v=q6E3xoKIBnY&list=PL-2EBeDYMIbQghmnb865lpdmYyWU3I5F1&index=3&ab_channel=BugBytes
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
