
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import google.generativeai as genai
from dotenv import load_dotenv
from maikol_utils.print_utils import print_separator, print_log
from maikol_utils.file_utils import save_json

from src.config import Configuration
from src.models import (
    MovieBatchClassification,
)
from src.models import classify_movies_batch
from src.scripts import compute_metrics


def predict_gemini(CONFIG: Configuration, validation: bool = False):
    # ========================================================
    #                       CONFIGURE MODEL
    # ========================================================
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    print_separator("CONFIGURE MODEL", sep_type="LONG")
    genai.configure(api_key=gemini_api_key)

    model = genai.GenerativeModel(
        'gemini-2.5-pro',
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=MovieBatchClassification
        )
    )
    # ========================================================
    #                       Load Data
    # ========================================================
    print_separator("LOADING DATA", sep_type="LONG")
    train_df = pd.read_csv(CONFIG.train_data)
    if validation:
        train_df_train = train_df.sample(frac=1-CONFIG.val_split, random_state=42)
        test_df = train_df.drop(train_df_train.index)
        train_df = train_df_train
    else:
        test_df = pd.read_csv(CONFIG.test_data)

    print_log(f" - Train shape: {train_df.shape}")
    print_log(f" - Test shape: {test_df.shape}")


    # ========================================================
    #                       PREDICT
    # ========================================================
    print_separator("PREDICTING", sep_type="LONG")
    predictions = classify_movies_batch(
        model,
        df_train=train_df,
        df_test=test_df,
        batch_size=CONFIG.batch_size
    )
    predictions_df = pd.DataFrame(
        [movie for movies in predictions for movie in movies["movies"]]
    )

    predictions_df.to_csv(
        CONFIG.gemini_results_path if validation else CONFIG.gemini_prediction_path, 
        index=False
    )
    
    if validation:
        # ========================================================
        #                       METRICS
        # ========================================================
        print_separator("METRICS", sep_type="LONG")
        train_df["labels"] = train_df["genre"].apply(lambda x: [l.strip() for l in x.split(",")])
        test_df["labels"] = test_df["genre"].apply(lambda x: [l.strip() for l in x.split(",")])

        # ========== Fit multilabel binarizer ========== 
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(train_df["labels"])

        # ========== Convert to binary format ========== 
        y_true_bin = mlb.transform(test_df["labels"].to_list())
        y_pred_bin = mlb.transform(predictions_df["genres"].to_list())

        # ========== Compute metrics ==========
        metrics = compute_metrics(y_true_bin, y_pred_bin)
        save_json(CONFIG.gemini_scores_path, metrics)

        for key, value in metrics.items():
            print_log(f" - {key:15}: {value:.4f}")