# import os
# import torch
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from PIL import Image, ImageFile
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision.transforms import Resize, ToTensor, Normalize, Compose
# from sklearn.preprocessing import MinMaxScaler
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#
# class ZeroShotDataset():
#     def __init__(self, data_df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len):
#         self.data_df = data_df
#         self.gtrends = gtrends
#         self.cat_dict = cat_dict
#         self.col_dict = col_dict
#         self.fab_dict = fab_dict
#         self.trend_len = trend_len
#         self.img_root = img_root
#
#     def __len__(self):
#         return len(self.data_df)
#
#     def __getitem__(self, idx):
#         return self.data_df.iloc[idx, :]
#
#     def preprocess_data(self):
#         data = self.data_df
#
#         # Get the Gtrends time series associated with each product
#         # Read the images (extracted image features) as well
#         gtrends, image_features = [], []
#         img_transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
#             cat, col, fab, fiq_attr, start_date, img_path = row['category'], row['color'], row['fabric'], row['extra'], \
#                 row['release_date'], row['image_path']
#
#             # Get the gtrend signal up to the previous year (52 weeks) of the release date
#             gtrend_start = start_date - pd.DateOffset(weeks=52)
#             cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[:self.trend_len]
#             col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[:self.trend_len]
#             fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[:self.trend_len]
#
#             cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
#             col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
#             fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
#             multitrends =  np.vstack([cat_gtrend, col_gtrend, fab_gtrend])
#
#
#             # Read images
#             img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')
#
#             # Append them to the lists
#             gtrends.append(multitrends)
#             image_features.append(img_transforms(img))
#
#         # Convert to numpy arrays
#         gtrends = np.array(gtrends)
#
#         # Remove non-numerical information
#         data.drop(['external_code', 'season', 'release_date', 'image_path'], axis=1, inplace=True)
#
#         # Create tensors for each part of the input/output
#         item_sales, temporal_features = torch.FloatTensor(data.iloc[:, :12].values), torch.FloatTensor(
#             data.iloc[:, 13:17].values)
#         categories, colors, fabrics = [self.cat_dict[val] for val in data.iloc[:].category.values], \
#                                        [self.col_dict[val] for val in data.iloc[:].color.values], \
#                                        [self.fab_dict[val] for val in data.iloc[:].fabric.values]
#
#
#         categories, colors, fabrics = torch.LongTensor(categories), torch.LongTensor(colors), torch.LongTensor(fabrics)
#         gtrends = torch.FloatTensor(gtrends)
#         images = torch.stack(image_features)
#
#         return TensorDataset(item_sales, categories, colors, fabrics, temporal_features, gtrends, images)
#
#     def get_loader(self, batch_size, train=True):
#         print('Starting dataset creation process...')
#         data_with_gtrends = self.preprocess_data()
#         data_loader = None
#         if train:
#             data_loader = DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=True, num_workers=0)
#         else:
#             data_loader = DataLoader(data_with_gtrends, batch_size=1, shuffle=False, num_workers=0)
#         print('Done.')
#
#         return data_loader
#

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazyDataset(Dataset):
    def __init__(self, item_sales, categories, colors, fabrics, temporal_features, gtrends, img_paths, img_root):
        self.item_sales = item_sales
        self.categories = categories
        self.colors = colors
        self.fabrics = fabrics
        self.temporal_features = temporal_features
        self.gtrends = gtrends
        self.img_paths = img_paths
        self.img_root = img_root
        self.transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.item_sales)

    def __getitem__(self, idx):
        # Load the image ONLY when the GPU asks for it during training
        img_path = self.img_paths[idx]
        img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')
        img_tensor = self.transforms(img)
        return (self.item_sales[idx], self.categories[idx], self.colors[idx], self.fabrics[idx],
                self.temporal_features[idx], self.gtrends[idx], img_tensor)

class ZeroShotDataset():
    def __init__(self, data_df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len):
        self.data_df = data_df
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_root = img_root

    def preprocess_data(self):
        data = self.data_df
        gtrends, image_paths = [], []

        for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
            cat, col, fab, fiq_attr, start_date, img_path = row['category'], row['color'], row['fabric'], row['extra'], \
                row['release_date'], row['image_path']

            gtrend_start = start_date - pd.DateOffset(weeks=52)
            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[:self.trend_len]
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[:self.trend_len]
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[:self.trend_len]

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
            col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
            fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
            multitrends = np.vstack([cat_gtrend, col_gtrend, fab_gtrend])

            gtrends.append(multitrends)
            image_paths.append(img_path) # Just save the path, not the actual image!

        gtrends = np.array(gtrends)

        data.drop(['external_code', 'season', 'release_date', 'image_path'], axis=1, inplace=True)

        item_sales, temporal_features = torch.FloatTensor(data.iloc[:, :12].values), torch.FloatTensor(
            data.iloc[:, 13:17].values)
        categories, colors, fabrics = [self.cat_dict[val] for val in data.iloc[:].category.values], \
                                       [self.col_dict[val] for val in data.iloc[:].color.values], \
                                       [self.fab_dict[val] for val in data.iloc[:].fabric.values]

        categories, colors, fabrics = torch.LongTensor(categories), torch.LongTensor(colors), torch.LongTensor(fabrics)
        gtrends = torch.FloatTensor(gtrends)

        return LazyDataset(item_sales, categories, colors, fabrics, temporal_features, gtrends, image_paths, self.img_root)

    def get_loader(self, batch_size, train=True):
        print('Starting dataset creation process...')
        data_with_gtrends = self.preprocess_data()
        if train:
            # We can safely use 2 workers now because we aren't duplicating 10GB of RAM!
            return DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=True, num_workers=2)
        else:
            return DataLoader(data_with_gtrends, batch_size=1, shuffle=False, num_workers=2)