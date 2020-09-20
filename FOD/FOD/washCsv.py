"""
序贯显示boudingbox。
csv里line编号显示在输出，记住上次line到哪下次输入在start_line里继续
"""
import os
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


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
# bbox is the abbreviation for bounding box

def lets_markLabels(data_dir,lines,start_line):
    imglist= sorted(glob.glob(os.path.join(data_dir, 'train_*.jpg')),key=lambda x:int(x.split('.jpg')[0][25:]))
    i=start_line
    for path_name in imglist:
        img=plt.imread(path_name)
        name=path_name.split("\\")[1]
        line=lines[i]
        import pdb
        #pdb.set_trace()
        while line[0]==name:
            temp=eval(line[5])
            try:
                bbox=[temp['x']-10,temp['y']-10,temp['x']+temp['width']+10,temp['y']+temp['weight']+10]
            except KeyError:
                print(name,'中无目标')
                i=i+1
                line=lines[i]  
                continue
            fig = plt.imshow(img)
            fig.axes.add_patch(bbox_to_rect(bbox, 'red'))
            plt.suptitle(name)
            plt.show()
            print("line NO:"+str(i))
            region_attributes=input()
            h_csv = HandleCsv('train_annotation.csv')
            h_csv.modify(i+1, 7, '{' + region_attributes + '}')
            print(h_csv.get_value(i+1,7))
            h_csv.list2csv('train_annotation.csv')
            i=i+1
            line=lines[i]
            
            #pdb.set_trace()
            #input("cintinue??")
            #plt.close()



if __name__=="__main__":
    start_line=1
    data_dir="./training_dataset"
    lines = read_csv_labels('train_annotation.csv')
    print(' len(lines)', len(lines))
    lets_markLabels(data_dir,lines,start_line)
    