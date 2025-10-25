"""Models.

Functions to manage, create, train / test models.
"""


from .gemini import (
    MovieGenre,
    MovieClassification,
    MovieBatchClassification,
    get_genres_list,
    generate_prompt,
    classify_with_model,
    classify_movies_batch,
    print_all_models
)