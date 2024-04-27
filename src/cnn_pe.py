
import os
from io import BytesIO
import boto3
import math
import logging
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image

from datasets import load_dataset
from datasets import Dataset

from dotenv import load_dotenv

from utils import label2numeric, EarlyStopping

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


load_dotenv()
AWS_BUCKET = os.getenv('AWS_BUCKET')
USE_AWS =os.getenv('USE_AWS', 'False').lower() == 'true'
LOCAL_DATA_PATH = os.getenv('LOCAL_DATA_PATH')
MODEL_PREFIX = os.getenv('MODEL_PREFIX')
HF_DATASET = os.getenv('HF_DATASET')
HF_DATASET_TRAIN_SIZE = int(os.getenv('HF_DATASET_TRAIN_SIZE'))
HF_DATASET_VALID_SIZE = int(os.getenv('HF_DATASET_VALID_SIZE'))
HF_DATASET_TEST_SIZE = int(os.getenv('HF_DATASET_TEST_SIZE'))


class BaseCNNPE:

    s3 = boto3.client('s3')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_map = {
        'PE and Others': 1,
        'PE Only': 1,
        'No Finding': 0
        }

    def __init__(self,
                 model_name: str,
                 model: nn.Module,
                 optimizer: optim,
                 criterion: nn.Module,
                 transform: transforms.Compose,
                 dataset_kwargs: dict = {'streaming': True},
                 label_map: dict = {'PE and Others': 1,
                                    'PE Only': 1,
                                    'No Finding': 0}
                 ) -> None:
        
        self.model_name = model_name

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.transform = transform
        self.dataset = load_dataset(HF_DATASET, **dataset_kwargs)
        self.label_map = label_map
        self.num_classes = len(set(self.label_map.values()))

    @classmethod
    def prepare_data(cls, dataset: Dataset, batch_size: int, 
                     label_map: dict, num_classes: int, 
                     transform: transforms.Compose) -> DataLoader:
        logger.info("Preparing the dataset")

        def map_labels(example):
            example['label'] = label2numeric(example['label'], label_map)
            return example


        def get_image(example):
            if USE_AWS:
                with BytesIO(cls.s3.get_object(Bucket=AWS_BUCKET,
                                                Key=example['file_path'])\
                            ['Body'].read()) as image_data_io:
                    image = Image.open(image_data_io)
            else:
                image = Image.open(os.path.join(LOCAL_DATA_PATH,
                                                example['file_path']))
                
            image = transform(image)

            label = label2numeric(example['label'], label_map)

            return image, label, example['dicom_id']


        def collate_get_image(batch):
            images = []
            labels = []
            dicom_ids = []

            for example in batch:

                image, label, dicom_id = get_image(example)
                images.append(image)
                labels.append(label)
                dicom_ids.append(dicom_id)
            # Stack the preprocessed images into a batch tensor
            images = torch.stack(images, dim=0)
            # labels = torch.tensor(labels).float()
            labels = nn.functional.one_hot(torch.tensor(labels),
                                           num_classes).float()
            # dicom_ids = torch.tensor(dicom_ids)
            return {
                'images': images,
                'labels': labels,
                'dicom_ids': dicom_ids
            }

        data_loader = DataLoader(dataset = dataset,
                                 batch_size = batch_size,
                                 collate_fn = collate_get_image)
        
        return data_loader
    
    def fit(self, n_epochs: int, start_epoch: int,
            batch_size: int, train_w_valid: bool = False):
        logger.info("Starting the training process")

        if train_w_valid:
            dataset = self.dataset['train'].concatenate(
                self.dataset['validation'])
            train_loader = self.prepare_data(dataset, batch_size, 
                                             self.label_map, self.num_classes, 
                                             self.transform)
            n_batches = math.ceil(
                (HF_DATASET_TRAIN_SIZE+HF_DATASET_VALID_SIZE)/batch_size)
        else:
            train_loader = self.prepare_data(self.dataset['train'],
                                             batch_size, self.label_map, 
                                             self.num_classes, self.transform)
            valid_loader = self.prepare_data(self.dataset['validation'],
                                             batch_size, self.label_map, 
                                             self.num_classes, self.transform)
            n_batches = math.ceil(HF_DATASET_TRAIN_SIZE/batch_size)
            
        # Training Loop
        for epoch in range(start_epoch, n_epochs+1):
            logger.info(f"EPOCH [{epoch}/{n_epochs}]")
            self.model.train()
            for i, batch in tqdm(enumerate(train_loader, 1)):
                images, labels = batch['images'], batch['labels']
                images, labels = images.to(self.device),\
                    labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Save a checkpoint
                if i%100 == 0 or i == n_batches:

                    logger.info(f'Epoch [{epoch}/{n_epochs}], '
                        f'Batch [{i}/{n_batches}], '
                        f'Train Loss: {loss.item():.4f}')

                    path = f"checkpoints/epoch_{str(epoch).zfill(2)}_batch_"\
                        f"{str(i).zfill(2)}.pth"
                    data = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                    }

                    _ = self._save_model(data, path)


            if not train_w_valid:

                val_loss, val_accuracy = self.validate(valid_loader)

                logger.info(f'Epoch [{epoch}/{n_epochs}], '
                    f'Train Loss: {loss.item():.4f}, '
                    f'Val Loss: {val_loss/HF_DATASET_VALID_SIZE:.4f}, '
                    f'Val Accuracy: {val_accuracy:.2f}%')


        # Save resulting model
        path = f"{self.model_name}-fully-trained.pth"
        model_path = self._save_model(self.model, path)
        
        data = {
            'n_epochs': n_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        self._save_model(data, f"{self.model_name}-fully-trained_metadata.pth")

        return model_path


    def validate(self, valid_loader: DataLoader):
        # Validation loop
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(valid_loader)):
                images, labels = batch['images'], batch['labels']
                images, labels = images.to(self.device),\
                    labels.to(self.device)
                labels_single = torch.argmax(labels, 1)
                outputs = self.model(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels_single).sum().item()
                val_loss += self.criterion(outputs, labels).item()

        val_accuracy = 100 * correct / total

        return val_loss, val_accuracy

    @classmethod
    def predict(cls, model: nn.Module, dataset: Dataset, 
                transform: transforms.Compose, batch_size: int, 
                label_map: dict = None) -> Tuple[list, list]:
        
        if label_map is None:
            label_map = cls.label_map

        num_classes = len(set(label_map.values()))

        test_loader = cls.prepare_data(dataset, batch_size, label_map, 
                                       num_classes, transform)

        model.eval()
        labels_list = []
        predicted_list = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader)):
                images, labels = batch['images'], batch['labels']
                images, labels = images.to(cls.device),\
                    labels.to(cls.device)
                labels_single = torch.argmax(labels, 1)
                outputs = model(images)
                predicted = torch.argmax(outputs, 1)

                labels_list += labels_single.tolist()
                predicted_list += predicted.tolist()

        return labels_list, predicted_list

    def _save_model(self, data, path):
        logger.info('Saving model data')

        full_path = f"{MODEL_PREFIX}/{self.model_name}/{path}"

        if USE_AWS:
            with BytesIO() as bytes:
                torch.save(data, bytes)
                bytes.seek(0)
                self.s3.put_object(Body=bytes,
                            Bucket=AWS_BUCKET,
                            Key=full_path)
                
            full_output_path = f"s3://{AWS_BUCKET}/{full_path}"

        else:
            full_output_path = os.path.join(LOCAL_DATA_PATH, full_path)
            dir2save = os.path.split(full_output_path)[0]
            if not os.path.exists(dir2save):
                os.makedirs(dir2save)
            torch.save(data, full_output_path)

        logger.info(f'Succesfully put data in {full_output_path}')
            
        return full_output_path
    

class GoogLeNetCNNPE(BaseCNNPE):

    def __init__(self,
            model_name: str,
            model: nn.Module,
            optimizer: optim,
            criterion: nn.Module,
            transform: transforms.Compose,
            early_stopper: EarlyStopping,
            lr_scheduler_kwargs: dict,
            dataset_kwargs: dict = {'streaming': True},
            label_map: dict = {'PE and Others': 1,
                            'PE Only': 1,
                            'No Finding': 0}
            ) -> None:
        
            super().__init__(model_name, model, optimizer, criterion,
                            transform, dataset_kwargs, label_map)
            self.early_stopper = early_stopper
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                                  **lr_scheduler_kwargs)

    def fit(self, n_epochs: int, start_epoch: int,
            batch_size: int, train_w_valid: bool = False):
        logger.info("Starting the training process")

        if train_w_valid:
            dataset = self.dataset['train'].concatenate(
                self.dataset['validation'])
            train_loader = self.prepare_data(dataset, batch_size)
            n_batches = math.ceil(
                (HF_DATASET_TRAIN_SIZE+HF_DATASET_VALID_SIZE)/batch_size)
        else:
            train_loader = self.prepare_data(self.dataset['train'],
                                                batch_size)
            valid_loader = self.prepare_data(self.dataset['validation'],
                                                batch_size)
            n_batches = math.ceil(HF_DATASET_TRAIN_SIZE/batch_size)
            
        # Training Loop
        for epoch in range(start_epoch, n_epochs+1):
            logger.info(f"EPOCH [{epoch}/{n_epochs}]")
            self.model.train()
            for i, batch in tqdm(enumerate(train_loader, 1)):
                images, labels = batch['images'], batch['labels']
                images, labels = images.to(self.device),\
                    labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Save a checkpoint
                if i%100 == 0 or i == n_batches:

                    logger.info(f'Epoch [{epoch}/{n_epochs}], '
                        f'Batch [{i}/{n_batches}], '
                        f'Train Loss: {loss.item():.4f}')

                    path = f"checkpoints/epoch_{epoch}_batch_{i}.pth"
                    data = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                    }

                    _ = self._save_model(data, path)


            self.lr_scheduler.step(loss)

            if not train_w_valid:

                val_loss, val_accuracy = self.validate(valid_loader)

                logger.info(f'Epoch [{epoch}/{n_epochs}], '
                    f'Train Loss: {loss.item():.4f}, '
                    f'Val Loss: {val_loss/HF_DATASET_VALID_SIZE:.4f}, '
                    f'Val Accuracy: {val_accuracy:.2f}%')
                
                self.early_stopper(val_loss)
                if self.early_stopper.early_stop:
                    logger.info("Early stopping triggered")
                    break

        # Save resulting model
        path = f"{self.model_name}-fully-trained.pth"
        model_path = self._save_model(self.model, path)

        data = {
            'n_epochs': n_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }

        self._save_model(data, f"{self.model_name}-fully-trained_metadata.pth")

        return model_path
