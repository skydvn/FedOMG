import os
import torch
from data.meta_dataset import MetaDataset, GetDataLoaderDict
from configs.default import vlcs_path
from torchvision import transforms

augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale( 0.1),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transform_test = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

vlcs_name_dict = {
    'v': 'voc2007',
    'l': 'labelme',
    'c': 'caltech101',
    's': 'sun09',
}

split_dict = {
    'train': 'train',
    'val': 'val',
    'total': 'test',
}


class VLCS_SingleDomain():
    def __init__(self, root_path=vlcs_path, domain_name='v', split='total', train_transform=None):
        if domain_name in vlcs_name_dict.keys():
            self.domain_name = vlcs_name_dict[domain_name]
            self.domain_label = list(vlcs_name_dict.keys()).index(domain_name)
        else:
            raise ValueError('domain_name should be in v l c s')
        
        self.root_path = os.path.join(root_path, 'images')
        self.split = split
        self.split_file = os.path.join(root_path, 'splits', f'{self.domain_name}_{split_dict[self.split]}' + '.txt')
        print(self.split_file)
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = transform_test
                
        imgs, labels = VLCS_SingleDomain.read_txt(self.split_file, self.root_path)
        self.dataset = MetaDataset(imgs, labels, self.domain_label, self.transform)
        
    @staticmethod
    def read_txt(txt_path, root_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            line_txt = line_txt.replace('\n', '')
            line_txt = line_txt.split(' ')
            imgs.append(os.path.join(root_path, line_txt[0]))
            labels.append(int(line_txt[1]))
        return imgs, labels
    
class VLCS_FedDG():
    def __init__(self, test_domain='v', batch_size=16):
        self.batch_size = batch_size
        self.domain_list = list(vlcs_name_dict.keys())
        print(self.domain_list)
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = VLCS_FedDG.SingleSite(domain_name, self.batch_size)
            
        
        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']
        
          
    @staticmethod
    def SingleSite(domain_name, batch_size=16):
        dataset_dict = {
            'train': VLCS_SingleDomain(domain_name=domain_name, split='train', train_transform=transform_train).dataset,
            'val': VLCS_SingleDomain(domain_name=domain_name, split='val').dataset,
            'test': VLCS_SingleDomain(domain_name=domain_name, split='total').dataset,
        }
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict, dataset_dict
        
    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict