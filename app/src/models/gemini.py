"""Code for Geminy model configuration and usage."""
import json
import enum
import pandas as pd
from tqdm import tqdm
from typing import List, TypedDict

import google.generativeai as genai

# Define the schema for structured output
class MovieGenre(enum.Enum):
    ACTION = "Action"
    ADVENTURE = "Adventure"
    ANIMATION = "Animation"
    COMEDY = "Comedy"
    CRIME = "Crime"
    DRAMA = "Drama"
    FAMILY = "Family"
    FANTASY = "Fantasy"
    HISTORY = "History"
    HORROR = "Horror"
    MUSIC = "Music"
    MYSTERY = "Mystery"
    ROMANCE = "Romance"
    SCIENCE_FICTION = "Science Fiction"
    TV_MOVIE = "TV Movie"
    THRILLER = "Thriller"
    WAR = "War"
    WESTERN = "Western"


class MovieClassification(TypedDict):
    movie_name: str
    genres: List[MovieGenre]

class MovieBatchClassification(TypedDict):
    movies: List[MovieClassification]

# ==============================================================
#                           Functions
# ==============================================================
def get_genres_list(df: pd.DataFrame) -> List[str]:
    genres = list(set([genre for genres in df["genre"].unique() for genre in genres.split(", ")]))
    genres_str = ", ".join(genres)
    return genres, genres_str

def generate_prompt(df_train: pd.DataFrame, df_batch: pd.DataFrame, genres_str: str) -> str:
    prompt = f"""
    Classify the following movie in any of these genres. More than one genre can be assigned.
    Genres: {genres_str}

    ==============================================
    """

    for idx, row in df_train.iterrows():
        prompt+=f"Movie title: {row['movie_name']}\n"
        prompt+=f"Plot: {row['description']}\n"
        prompt+=f"Actual genres: {row['genre']}\n"
        prompt+="-------------------\n"
    
    prompt += """
    ==============================================
    Now, classify the following movies returning a structured JSON response with the movie names and their genres.
    """

    for idx, row in df_batch.iterrows():
        prompt += f"Movie title: {row['movie_name']}\n"
        prompt += f"Plot: {row['description']}\n"
        prompt += "\n"
    
    return prompt

def print_all_models():
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(model.name)

# ==============================================================
#                           CLASSIFY
# ==============================================================
def classify_with_model(
    model: genai.GenerativeModel, 
    prompt: str
) -> List[MovieClassification]: 
    """Classify movies using structured output"""
    
    response = model.generate_content(prompt)
    batch_result = json.loads(response.text)
    return batch_result


def classify_movies_batch(
    model: genai.GenerativeModel,
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    batch_size: int,
):
    """Classify movies in batches using structured output"""
    results = []
    genres, genres_str = get_genres_list(df_train)

    for i in tqdm(range(0, len(df_test), batch_size)):
        batch = df_test.iloc[i:i+batch_size]

        try:
            prompt = generate_prompt(df_train, batch, genres_str)
            results.append((
                batch["movie_name"].values,
                batch["description"].values,
                classify_with_model(model, prompt)
            ))

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
    
    return results