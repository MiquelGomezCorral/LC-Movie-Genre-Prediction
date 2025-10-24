"""Main file for scripts with arguments and call other functions."""

import argparse
import dotenv
from maikol_utils.other_utils import args_to_dataclass
from src.config import Configuration
from src.scripts import example

def cmd_read_extract(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    ...

def cmd_train(args):
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    example(CONFIG)

def cmd_test(args):
    """Call test functions."""
    example()

# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       read_extract
    # ======================================================================================
    p_read = subparsers.add_parser("read-extract", help="Read and extract from config list")
    p_read.add_argument(
        "-d", "--dataset_name", type=str, default="Nuelas", help="Name of raw data folder"
    )
    p_read.add_argument("-m", "--max_files", type=int, default=None, help="Max files to load")
    p_read.add_argument(
        "-l", "--use_llm", action="store_false", default=True, help="Disable LLM extraction"
    )
    p_read.set_defaults(func=cmd_read_extract)

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