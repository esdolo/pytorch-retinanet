import os
import json
import pandas as pd

import csv
import glob
from matplotlib import pyplot as plt

class HandleCsv:
    # 定义存放csv内容的list
    csv_list = []

    def __init__(self, filename):
        self.filename = filename
        with open(self.filename)as fp:
            self.csv_list = list(csv.reader(fp))
            #print(self.csv_list)

    # 在第N行第M列空白单元格处修改内容
    def modify(self, n, m, value):
        self.csv_list[n - 1][m - 1] = value

    def get_value(self, n, m):
        return self.csv_list[n - 1][m - 1]

    def list2csv(self, file_path):
        try:
            fp = open(file_path, 'w')
            for items in self.csv_list:
                for i in range(len(items)):
                    if items[i].find(',') != -1:
                        fp.write('\"')
                        fp.write(items[i])
                        fp.write('\"')
                    else:
                        fp.write(items[i])
                    if i < len(items) - 1:
                        fp.write(',')
                fp.write('\n')
        except Exception as e:
            print(e)



def read_csv_labels(fname):
    """Read fname.csv to return the lines."""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        #lines = f.readlines()[1:]
        lines=[]
        reader = csv.reader(f)
        for row in reader:
            lines.append(row)
    f.close()
    return lines

def main():
    lines = read_csv_labels('train_annotation.csv')
    result = "./train_standard_annotation.csv"
    tempDF = pd.DataFrame(columns=["path", "xmin", "ymin", "xmax", "ymax", "class"])
    imgpath='./VOD/JPEGImages/'
    #import pdb
    #pdb.set_trace()
    for line in lines[1:]:
        imgname=line[0]
        linexys=eval(line[5])
        temp = {}
        temp["path"]=imgpath+imgname
        temp["xmin"] = linexys['x']
        temp["ymin"] = linexys['y']
        temp["xmax"] = linexys['width']+linexys['x']
        temp["ymax"] = linexys['height']+linexys['y']
        temp["class"] = line[6]
        tempDF = tempDF.append(temp, ignore_index=True)
    tempDF.to_csv(result, index=False)


def convert(x, y):
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    return min_x, min_y, max_x - min_x, max_y - min_y


if __name__ == "__main__":
    main()
