"""Main file for scripts with arguments and call other functions."""

import argparse
import dotenv
from maikol_utils.other_utils import args_to_dataclass
from src.config import Configuration
from src.data import prepare_data_train
from src.scripts import train, predict_and_metrics

def cmd_train(args):
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    CONFIG.columns = ["movie_name","description"]
    X_train, X_val, y_train, y_val = prepare_data_train(path = CONFIG.train_data, bag_X=CONFIG.columns, label=CONFIG.label)

    model = train(X_train, y_train)
    metrics = predict_and_metrics(model, X_val, y_val)
    print(metrics)

def cmd_test(args):
    """Call test functions."""
    pass

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
    p_train = subparsers.add_parser("train", help="Train script with any code")
    p_train.add_argument(
        "-r", "--run_name", type=str, default=None, help="Name of run"
    )
    p_train.set_defaults(func=cmd_train)


    # ======================================================================================
    #                                       test
    # ======================================================================================
    p_test = subparsers.add_parser("test", help="Test script with any code")
    p_test.set_defaults(func=cmd_test)

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)