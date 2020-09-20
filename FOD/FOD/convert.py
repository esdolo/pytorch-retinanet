import os
import json
import pandas as pd


def main():
    filename = "./via_region_data.json"
    result = "./train_annotation.csv"
    with open(filename, "r") as f:
        json_dict = json.load(f)
    tempDF = pd.DataFrame(columns=["#filename", "file_size", "file_attributes", "region_count", "region_id", "region_shape_attributes", "region_attributes"])
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
                temp_dict = {}
                temp["#filename"] = "train_" + str(count + i) + ".jpg"
                temp["file_size"] = os.path.getsize("./training_dataset/" + temp["#filename"])
                temp["file_attributes"] = temp_dict
                temp["region_count"] = 0
                temp["region_id"] = 0
                temp["region_shape_attributes"] = temp_dict
                temp["region_attributes"] = temp_dict
                tempDF = tempDF.append(temp, ignore_index=True)
            count = idx
        count += 1
        for region in regions:
            temp = {}
            temp["#filename"] = picName
            temp["file_size"] = item["size"]
            temp["file_attributes"] = item["file_attributes"]
            temp["region_count"] = len(item["regions"])
            temp["region_id"] = int(region)
            x, y, width, height = convert(regions[region]["shape_attributes"]["all_points_x"], regions[region]["shape_attributes"]["all_points_y"])
            temp_dict = {}
            temp_dict["name"] = "rect"
            temp_dict["x"] = x
            temp_dict["y"] = y
            temp_dict["width"] = width
            temp_dict["weight"] = height
            temp["region_shape_attributes"] = temp_dict
            temp_dict = {}
            temp["region_attributes"] = temp_dict
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
