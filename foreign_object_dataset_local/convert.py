import os
import pandas as pd
from xml.dom.minidom import parse


class Lable:

    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height

    def __eq__(self, o: object) -> bool:
        if type(self) == type(o) and self.ymin == o.ymin and self.xmin == o.xmin and self.width == o.width and self.height == o.height:
            return True
        else:
            return False


def main():
    result = "./train_annotation.csv"
    resultDF = pd.DataFrame(columns=["#filename", "file_size", "file_attributes", "region_count", "region_id", "region_shape_attributes", "region_attributes"])
    
    annotations_dir = "./Annotations"
    images_dir = "./JPEGImages"

    filelist = os.listdir(annotations_dir)
    filelist = sorted(filelist, key=lambda x:int(x.split('.')[0]))

    for annotation_name in filelist:
        domTree = parse(os.path.join(annotations_dir, annotation_name))
        rootNode = domTree.documentElement
        # filename
        filenames = rootNode.getElementsByTagName("filename")
        filename = filenames[0].childNodes[0].nodeValue
        print("filename: ", filename)
        # filesize
        filesize = os.path.getsize(os.path.join(images_dir, filename))
        print("filesize: ", filesize)
        # file_attributes
        file_attributes = {}
        print("file_attributes: ", file_attributes)
        # objects
        objects = rootNode.getElementsByTagName("object")
        # region_count
        region_count = len(objects)
        print("region_count: ", region_count)
        # object
        lable_list = []
        for object in objects:
            # region_id
            bndbox = object.getElementsByTagName("bndbox")[0]
            xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].nodeValue)
            xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].nodeValue)
            ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].nodeValue)
            ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].nodeValue)
            print("xmin: ", xmin)
            print("xmax: ", xmax)
            print("ymin: ", ymin)
            print("ymax: ", ymax)
            
            lable = Lable(xmin, ymin, xmax - xmin, ymax - ymin)
            flag = True
            for item in lable_list:
                if lable.__eq__(item):
                    flag = False
                    break
            if flag:
                lable_list.append(lable)
        for i in range(len(lable_list)):
            # region_shape_attributes
            region_shape_attributes = {}
            region_shape_attributes["name"] = "rect"
            region_shape_attributes["x"] = lable_list[i].xmin
            region_shape_attributes["y"] = lable_list[i].ymin
            region_shape_attributes["width"] = lable_list[i].width
            region_shape_attributes["height"] = lable_list[i].height
            print("region_shape_attributes: ", region_shape_attributes)
            # region_attributes
            region_attributes = {}
            print("region_attributes: ", region_attributes)

            temp_dict = {}
            temp_dict["#filename"] = filename
            temp_dict["file_size"] = filesize
            temp_dict["file_attributes"] = file_attributes
            temp_dict["region_count"] = len(lable_list)
            temp_dict["region_id"] = i
            temp_dict["region_shape_attributes"] = region_shape_attributes
            temp_dict["region_attributes"] = region_attributes
            resultDF = resultDF.append(temp_dict, ignore_index=True)
            """
            print("resultDF: ")
            print(resultDF)
            input()"""

    resultDF.to_csv(result, index=False)


if __name__ == "__main__":
    main()
