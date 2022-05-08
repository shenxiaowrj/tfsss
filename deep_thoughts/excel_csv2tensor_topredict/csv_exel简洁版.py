# _*_coding:utf-8_*_
# 2022-05-07 ~ 2022-05-08
# shenxiao
"""演示如何从Excel/CSV文件中读取数据转成PyTorch Tensor用于机器学习模型训练"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas

class ExcelDataset(Dataset):
    def __init__(self, filepath=" ", sheet_name=0):      #怎样在这里设置数据格式的提示啊？？？

        print(f"reading {filepath}, sheet={sheet_name}")

        df = pandas.read_excel(
            filepath,
            header=0,
            index_col=0,
            names=['feat1','feat2','label'],
            sheet_name=sheet_name,
            dtype={"feat1": np.float32,
                   "feat2": np.float32,
                   "label": np.int32}
        )

        print(f"the shape of dataframe is {df.shape}")

        feat = df.iloc[:, :2].values
        label = df.iloc[:, 2].values

        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class CsvDataset(Dataset):

    def __init__(self, filepath="data.csv"):
        # there is no sheet name definition in csv format file

        print(f"reading {filepath}")

        df = pandas.read_csv(
            filepath,
            header=0,
            index_col=0,
            encoding = 'utf-8',
            names= ['feat1', 'feat2', 'label'],
            dtype = {"feat1": np.float32, "feat2": np.float32, "label": np.int32},
            skip_blank_lines = True,
        )

        feat = df.iloc[:, :2].values
        label = df.iloc[:, 2].values

        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class Csv2Dataset(Dataset):

    def __init__(self, filepath="data.csv"):
        # there is no sheet name definition in csv format file:

        print(f"reading {filepath}")

        with open(filepath, encoding='utf-8') as f:
            lines = f.readlines()

        feat = []
        label = []
        for line in lines[1:]:
            values = line.strip().split(',')
            row_feat = [float(v) for v in values[1:3]]
            row_label = int(values[3])

            feat.append(row_feat)
            label.append(row_label)

        feat = np.array(feat, dtype=np.float32)
        label = np.array(label, dtype=np.int32)

        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

if __name__ == '__main__':

    print("Test for ExcelDataset")
    exel_dataset = ExcelDataset(filepath="data.xlsx",sheet_name="corpus1")
    # exel_dataset = ExcelDataset(sheet_name="corpus2")
    # exel_dataset = ExcelDataset(sheet_name=None)
    exel_dataloader = DataLoader(exel_dataset, batch_size=8, shuffle=True)
    for idx, (batch_x, batch_y) in enumerate(exel_dataloader):
        print(f"batch_id: {idx}, {batch_x.shape}, {batch_y.shape}")
        print(batch_x,batch_y)

    print("Test for CsvDataset")
    csv_dataset = CsvDataset()
    csv_dataloader = DataLoader(csv_dataset,batch_size=8,shuffle=True)
    for idx, (batch_x,batch_y) in enumerate(csv_dataloader):
        print(f"batch_id:{idx},{batch_x.shape},{batch_y.shape}")
        print(batch_x,batch_y)


    print("Test for Csv2Dataset")
    csv_dataset = Csv2Dataset()
    csv_dataloader = DataLoader(csv_dataset,batch_size=8,shuffle=True)
    for idx, (batch_x,batch_y) in enumerate(csv_dataloader):
        print(f"batch_id:{idx},{batch_x.shape},{batch_y.shape}")
        print(batch_x,batch_y)