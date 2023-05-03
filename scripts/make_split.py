import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process scrapped json.')
    parser.add_argument('--path_to_csv', type=str, help="Full path to processed csv.", default="processed_news.csv")
    parser.add_argument('--pandas_csv_sep', type=str, help="Separator of csv.", default=",")
    parser.add_argument('--test_size', type=float, help="example: 0.1", default=0.0)
    parser.add_argument('--val_size', type=float, help="example: 0.1", default=0.0)
    parser.add_argument('--train_path', type=str, help="Full path to processed csv.", default="train_split.csv")
    parser.add_argument('--test_path', type=str, help="Full path to processed csv.", default="test_split.csv")
    parser.add_argument('--val_path', type=str, help="Full path to processed csv.", default="val_split.csv")

    args = dict(vars(parser.parse_args()))
    if args['test_size'] == 0.0 and args['val_size'] > 0.0:
        raise Exception('if test part is not needed, use --test_size <needed val size>, use --val_path val_split.csv')

    df = pd.read_csv(args['path_to_csv'], sep=args['pandas_csv_sep'])

    train_df, test_df = train_test_split(df, test_size=args['test_size'], shuffle=True)
    if args['val_size'] > 0.0:
        train_df, val_df = train_test_split(train_df, test_size=args['val_size'] / (1 - args['test_size']), shuffle=True)
    else:
        val_df = None

    train_df.to_csv(args['train_path'], index=False)
    test_df.to_csv(args['test_path'], index=False)
    if val_df is not None:
        val_df.to_csv(args['val_path'], index=False)
