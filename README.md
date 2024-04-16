# pleural-effusion-cnn
*Dalton Macres, Jeffrey Chan, Jingni Cai*\
*Spring 2024*

Predict plueral effusion in chest x-rays using the MIMIC-CXR-JPG dataset. This project was done for CS534 Artificial Intelligence course of WPI's masters of data science program during the Spring 2024 semester.

## Repo Structure
** *subject to change* **
- **data**/ - any small size data files related to the project
- **src**/
    - **architectures.py** - SOTA CNN architecture implementations
    - **base_cnn_pe.py** - a base class for fitting and predicting with any model architecture
    - **train.py** - main code that trains a CNN
- **test**/ - various testing code
- **utils**/
    - **prep_data**/ - python files that prepare data and do a train-valid-test split
    - **utils**/ - various utility methods used in the modeling efforts

## Instructions
### Training your model
[Python's click](https://click.palletsprojects.com/en/8.1.x/) module was used for easy CLI interaction.
For instance, if you would like to train the customized resnet model, you would run the following from the root directory:

```python
python src/train.py resnet
```


### How to Implement the BaseCNNPE Class
This is a framework class that can be used to train a CNN of any architecture as well as make predictions on data using a trained model. There are two options to use this class:

1. Create a **BaseCNNPE** class object for a model architecture
    - If the preprocessing and training methods are sufficient for your architecture, this option would be ideal.
    - Below is an example of how to implement this object and train a model

        ```python
            from base_cnn_pe import BaseCNNPE

            model_obj = BaseCNNPE(
                model_name,
                model,
                optimizer,
                criterion,
                transform
            )

            trained_model_s3_path = model_obj.fit(n_epochs, batch_size, train_w_valid)
        ```

2. Create a sub-class of **BaseCNNPE**
    - This method should be used when a specific model architecture requires a different preprocessing or fit method

### Environment Setup Instructions
#### Conda Environment Setup
1. Install [anaconda](https://www.anaconda.com/download)
2. Change directory to the root of this repository
    ```bash
    cd <PATH/TO/LOCAL/pleural-effusion-cnn>
    ```
3. Create the environment
    ```bash
    conda create -n cnn-pe python=3.10
    ```
4. Activate the environment
    ```bash
    conda activate cnn-pe
    ```
5. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```
6. Set conda to develop mode
    ```bash
    conda develop .
    ```

#### AWS Key Setup
1. The AWS CLI should have been installed with the requirements above. Configure your keys by running:
    ```bash
    aws configure
    ```
2. Enter your access key provided by the account owner
3. Enter your secret access key provided by the account owner
4. Enter us-east-1 for the default region name
5. Enter json for the default output format

#### HuggingFace Setup
This is only needed if using private datasets/models. As of now (03/30/2024), the dataset is public, so this setup can be skipped. Instructions are provided here for future reference.

1. Create or log in to your [HuggingFace](https://huggingface.co/) account
2. Create an [access token](https://huggingface.co/docs/hub/en/security-tokens)
3. The huggingface-cli should have already been installed from the requirements above. Log in to your account by running:
    ```bash
    huggingface-cli login
    ```
4. Enter the access token you just created
