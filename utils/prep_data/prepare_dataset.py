import os
import argparse
import pandas as pd
import subprocess

WGET_BASE_COMMAND = 'wget -r -N -c -np --user={} --password={} {}'
MIMIC_CXR_JPG_URL = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0/'
S3_BUCKET = 'mimic-cxr-jpg-cs534-team11'
S3_PATH = 'data'


def execute_bash_command(bash_command):
    """
    Executes a bash commandfrom Python (requires Linux based terminal)
    """

    try:
        subprocess.run(bash_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error executing bash command: {e}')
        

def remove_directory(dir):
    """
    Removes a directory
    """

    execute_bash_command(f'rm -rf {dir}')


def get_pe_and_normal_dataset(df):
    """
    Subsets a dataframe of labels on Pleural Effusion (PE) and "Normal" (No
    Findings)
    """

    def label_data(row):
        """
        Nested function to label PE and No Findings
        """

        if row['Pleural Effusion'] == 1:
            if row['N_Findings'] == 1:
                return 'PE Only'
            elif row['N_Findings'] > 1:
                return 'PE and Others'
        elif row['No Finding'] == 1:
            return 'No Finding'
        
        return None
    
    cols2use = [col for col in df.columns if col not in ['subject_id', 'study_id']]

    df['N_Findings'] = df[cols2use].sum(axis=1)

    df['label'] = df.apply(label_data, axis=1)

    return df[df.label.isin(['No Finding', 'PE Only', 'PE and Others'])].\
        reset_index(drop=True)


def filter_metadata(df, view):
    """
    Subset the metadata based on the CXR ViewPosition, and the most common
    CXR dimensions (Rows, Columns).
    """

    rows, cols = df[df.ViewPosition == view][['Rows', 'Columns']]\
        .value_counts().index[0]
    
    return df[(df.ViewPosition == view) & 
              (df.Rows == rows) &
              (df.Columns == cols)].reset_index(drop=True)


def main(args):

    if not os.path.exists('physionet.org'):
        # Download the main files (non-JPG files) from MIMIC-CXR-JPG
        get_base_files_command = WGET_BASE_COMMAND\
            .format(args.user,
                    args.password,
                    f"--exclude-directories=files/mimic-cxr-jpg/2.1.0/files \
                        {MIMIC_CXR_JPG_URL}")
        execute_bash_command(get_base_files_command)

    # Decompress the GZip files
    local_dir = MIMIC_CXR_JPG_URL.split('//')[1]
    ungzip_command = f'for file in {local_dir}*.gz; do gzip -d "$file"; done'
    execute_bash_command(ungzip_command)

    # Get PE and Normal subset
    base_label_file_name = f'mimic-cxr-2.0.0-{args.labeler}'
    labels = pd.read_csv(f'{local_dir}{base_label_file_name}.csv')
    labels = get_pe_and_normal_dataset(labels)

    # Filter metadata based on view
    metadata = pd.read_csv(f'{local_dir}mimic-cxr-2.0.0-metadata.csv',
                           usecols = ['dicom_id', 'subject_id', 'study_id',
                                      'ViewPosition', 'Rows', 'Columns'])
    metadata = filter_metadata(metadata, args.cxr_view)

    # Join with Metadata and only get the Antero-Posterior (AP) images
    df = pd.merge(labels, metadata, how = 'inner',
                  on = ['subject_id', 'study_id'])
    # print(df.label.value_counts())

    # Save the file
    df.to_csv(f'{local_dir}{base_label_file_name}-PE-SUBSET.csv')

    # Upload main files to s3
    upload_command_format = 'aws s3 cp {} {} --recursive'
    upload_main_command = upload_command_format.format(local_dir,
                                                       f's3://{S3_BUCKET}/'+\
                                                        f'{S3_PATH}/')
    execute_bash_command(upload_main_command)

    # Download the image subset and upload to s3
    download_image_command_format =\
    'grep "{}" {}IMAGE_FILENAMES | wget -r -N -c -np -nH --cut-dirs=1 ' +\
        '--user={} --password={} -i - --base={}'
    
    cxr_dir = 'mimic-cxr-jpg'
    upload_cxr_command = upload_command_format.format(cxr_dir,
                                                    f's3://{S3_BUCKET}/'+\
                                                    f'{S3_PATH}/{cxr_dir}')
    
    for idx, dicom_id in enumerate(df.dicom_id.tolist()):

        download_image_command = download_image_command_format.format(
            dicom_id, local_dir, args.user, args.password, MIMIC_CXR_JPG_URL
        )
        execute_bash_command(download_image_command)

        # Upload then locally delete every 10th iteration
        if idx%10==0:
            execute_bash_command(upload_cxr_command)
            remove_directory(cxr_dir)

    execute_bash_command(upload_cxr_command)
    remove_directory(cxr_dir)
    remove_directory(local_dir)


if __name__ == '__main__':

    if not os.path.exists('./data'):
        os.makedirs('./data')

    os.chdir('./data')
    print(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default = 'dmacres', help=
                        "PhysioNet Username")
    parser.add_argument("--password", type=str, help="PhysioNet Password",
                        required = True)
    parser.add_argument("--labeler", type=str, default = 'chexpert',
                        help="The labeler file to use (chexpert or negbio)")
    parser.add_argument("--cxr-view", type=str, default = 'AP',
                        help="The CXR view (AP, PA, or Lateral)")

    args = parser.parse_args()

    main(args)
