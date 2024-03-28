
import os
from io import StringIO
import pandas as pd
import boto3

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


s3 = boto3.client('s3')


def make_paths_df(path, download=False):
    """
    Downloads (if download=True) the IMAGE_FILENAMES dataset and converts it
    to a dataframe with s3 paths and dicom id.
    """

    if download:
        s3.download_file('mimic-cxr-jpg-cs534-team11', 'data/IMAGE_FILENAMES', path)

    with open('data/IMAGE_FILENAMES', 'rb') as file:
        data = file.read()

    data_txt = data.decode('utf-8')

    df = pd.read_csv(StringIO(data_txt), header=None, names = ['file_path'])

    df['file_path'] = df.file_path.apply(lambda x: os.path.join('data/mimic-cxr-jpg/2.1.0', x))
    df['dicom_id'] = df.file_path.apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    return df


def prep_data(df):
    """
    Preps data for huggingface dataset. Keeps only a subset of key features and
    puts the rest in 'metadata'.
    """

    metadata_cols = [col for col in df.columns if col not in ['dicom_id', 'file_path', 'label']]

    df['metadata'] = df.apply(lambda x: {col: x[col] for col in metadata_cols}, axis = 1)

    df.drop(columns = metadata_cols, inplace = True)


def main():
    s3_uri = 's3://mimic-cxr-jpg-cs534-team11/data/mimic-cxr-2.0.0-chexpert-PE-SUBSET.csv'
    sub = pd.read_csv(s3_uri).drop(columns = ['Unnamed: 0'])

    paths = make_paths_df('data/IMAGE_FILENAMES')

    df = pd.merge(sub, paths, how = 'inner', on = 'dicom_id')

    prep_data(df)

    train_valid_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    train_df, valid_df = train_test_split(train_valid_df, test_size=0.25, random_state=123)

    train = Dataset.from_pandas(train_df)
    valid = Dataset.from_pandas(valid_df)
    test = Dataset.from_pandas(test_df)

    ds = DatasetDict({'train': train, 'validation': valid, 'test': test})
    ds.push_to_hub(repo_id = 'dmacres/cnn-pe', token = True)



if __name__ == '__main__':
    main()
