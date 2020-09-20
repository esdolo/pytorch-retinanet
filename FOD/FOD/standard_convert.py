import os
import json
import pandas as pd


def main():
    filename = "./via_region_data.json"
    result = "./train_standard_annotation.csv"
    with open(filename, "r") as f:
        json_dict = json.load(f)
    tempDF = pd.DataFrame(columns=["path", "xmin", "ymin", "xmax", "ymax", "class"])
    count = 1
    #import pdb
    #pdb.set_trace()
    sorted_keys=sorted(json_dict.keys(),key=lambda x:int(x.split('.')[0]))
    for key in sorted_keys:
        item = json_dict[key]
        regions = item["regions"]
        idx = int(key.split(".")[0])
        picName = "train_" + str(idx) + ".jpg"
        if idx != count:
            for i in range(idx - count):
                temp = {}
                temp["path"] = "./FOD/FOD/training_dataset/train_" + str(count + i) + ".jpg"
                temp["xmin"] = None
                temp["xmax"] = None
                temp["ymin"] = None
                temp["ymax"] = None
                temp["class"] = None
                tempDF = tempDF.append(temp, ignore_index=True)
            count = idx
        count += 1
        for region in regions:
            temp = {}
            temp["path"] = "./FOD/FOD/training_dataset/" + picName
            x, y, width, height = convert(regions[region]["shape_attributes"]["all_points_x"], regions[region]["shape_attributes"]["all_points_y"])
            if width==0:
                continue
            temp["xmin"] = x
            temp["ymin"] = y
            temp["xmax"] = x+width
            temp["ymax"] = y+height
            temp["class"] = "UnDefined"
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
