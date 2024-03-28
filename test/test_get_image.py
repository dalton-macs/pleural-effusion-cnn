from utils.utils.data import map_labels, collate_get_image
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm


label_converter = {'PE and Others': 1, 'PE Only': 1, 'No Finding': 0}
dataset = load_dataset('dmacres/cnn-pe', split = 'train', streaming=True)
dataset = dataset.map(lambda x: map_labels(x, label_converter))
# print(dataset[0])

batch_size = 32
data_loader = DataLoader(dataset,
                         batch_size=batch_size,
                         collate_fn=collate_get_image)

for i, batch in tqdm(enumerate(data_loader)):
    images = batch['images']
    labels = batch['labels']
    # dicom_ids = batch['dicom_ids']
    print(labels)