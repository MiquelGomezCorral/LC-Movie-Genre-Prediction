"""Main file for scripts with arguments and call other functions."""
import pandas as pd
import os
import argparse
import dotenv
from maikol_utils.other_utils import args_to_dataclass
from src.config import Configuration
from src.data import prepare_data_train, prepare_data_train_transformer, prepare_data_test, prepare_data_test_transformer
from src.scripts import test, train, predict_and_metrics, predict_gemini
from src.models import TransformerMultiLabelTrainer

def cmd_train(args):
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    CONFIG.columns = ["movie_name","description"]

    X_train, X_val, y_train, y_val = prepare_data_train(CONFIG)

    model = train(X_train, y_train)
    metrics = predict_and_metrics(model, X_val, y_val)
    print(metrics)

def cmd_train_transformer(args):
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    CONFIG.columns = ["movie_name","description"]

    train_dataset, val_dataset = prepare_data_train_transformer(CONFIG)
    model_name = "results/last_model"
    tokenizer_name = "FacebookAI/roberta-large-mnli"
    model = TransformerMultiLabelTrainer(model_name, tokenizer_name)

    model.train(train_dataset, val_dataset)

def cmd_predict_gemini(args):
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    predict_gemini(CONFIG, args.validation)

def cmd_test(args):
    """Call test functions."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    CONFIG.columns = ["movie_name","description"]

    X_test = prepare_data_test(CONFIG)
    y_pred = test(X_test)
    
    df = pd.read_csv(CONFIG.test_data)
    df["genre"] = y_pred
    os.makedirs("RL", exist_ok=True)  
    df.to_csv("RL/regresion_logistica.csv", index=False)
    
def cmd_test_transformers(args):
    """Call test functions."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    CONFIG.columns = ["movie_name","description"]

    X_test = prepare_data_test_transformer(CONFIG)
    
    model_name = "../results/roberta667/best_model"
    tokenizer_name = "FacebookAI/roberta-large-mnli"
    model = TransformerMultiLabelTrainer(model_name, tokenizer_name)

    y_pred = model.test(X_test)

    df = pd.read_csv(CONFIG.test_data)
    df["genre"] = y_pred
    os.makedirs("transformers", exist_ok=True)
    df.to_csv("transformers/transformers.csv", index=False)


# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       train
    # ======================================================================================
    p_train = subparsers.add_parser("train", help="Train script classic models")
    p_train.add_argument(
        "-t", "--transformer", type=str, default=None, help="Name of run"
    )
    p_train.set_defaults(func=cmd_train)

    # ======================================================================================
    #                                       train_transformer
    # ======================================================================================
    p_transformer = subparsers.add_parser("transformer", help="Train transformers")
    p_transformer.set_defaults(func=cmd_train_transformer)

    # ======================================================================================
    #                                       gemini
    # ======================================================================================
    p_gemini = subparsers.add_parser("gemini", help="Gemini script to predict test")
    p_gemini.add_argument(
        "-v", "--validation", default=False, action="store_true", 
        help="Predict for validation set instead of test set"
    )
    p_gemini.set_defaults(func=cmd_predict_gemini)
    
    # ======================================================================================
    #                                       test
    # ======================================================================================
    p_test = subparsers.add_parser("test", help="Test script classic models")
    p_test.set_defaults(func=cmd_test)

    # ======================================================================================
    #                                       test_transformer
    # ======================================================================================
    p_test = subparsers.add_parser("test_t", help="Test transformers")
    p_test.set_defaults(func=cmd_test_transformers)
    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)
