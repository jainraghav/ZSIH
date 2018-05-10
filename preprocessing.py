import os
import pdb
from PIL import Image
import pandas as pd
from numpy import array

def preprocess_img_folder(IMG_PATH,ALL_DATA):

    all_image_classes = os.listdir(IMG_PATH)
    all_image_path_set = []

    for cl in all_image_classes:
        if os.path.isdir(IMG_PATH + cl + '/'):
            class_files = os.listdir(IMG_PATH + cl + '/')
            refined_class_files = [cl+'/'+x for x in class_files]
            all_image_path_set = all_image_path_set + refined_class_files

    with open(ALL_DATA, "w") as f1:
        f1.write("ImagePath\n")
        for line in all_image_path_set:
            f1.write(line+"\n")

#for zero-shot learning task
def divide_into_sets_disjointclasses(ALL_DATA,IMG_PATH,trainp=0.6,validp=0.2,testp=0.2):

    tmp_df = pd.read_csv(ALL_DATA)
    arr = tmp_df['ImagePath'].str.partition('/')[0].values.tolist()
    all_classes = set(arr)
    all = list(all_classes)
    train_set = all[:int(trainp*len(all))]
    valid_set = all[int(trainp*len(all))+1:int(trainp*len(all))+1+int(validp*len(all))]
    test_set = all[int(trainp*len(all))+1+int(validp*len(all)):]

    TRAIN_DATA = IMG_PATH + "filelist-train.txt"
    with open(ALL_DATA) as f:
        with open(TRAIN_DATA, "w") as f1:
            f1.write("ImagePath\n")
            for line in f:
                if line.split('/')[0] in train_set:
                    f1.write(line)

    TEST_DATA = IMG_PATH + "filelist-test.txt"
    with open(ALL_DATA) as f:
        with open(TEST_DATA, "w") as f1:
            f1.write("ImagePath\n")
            for line in f:
                if line.split('/')[0] in test_set:
                    f1.write(line)

    VALID_DATA = IMG_PATH + "filelist-valid.txt"
    with open(ALL_DATA) as f:
        with open(VALID_DATA, "w") as f1:
            f1.write("ImagePath\n")
            for line in f:
                if line.split('/')[0] in valid_set:
                    f1.write(line)

    return TRAIN_DATA,TEST_DATA,VALID_DATA

#for normal classification learning task
def divide_into_sets_allclasses(ALL_DATA,IMG_PATH,trainp=0.6,validp=0.2,testp=0.2):
    train_set=[]
    valid_set=[]
    test_set=[]
    tmp_df = pd.read_csv(ALL_DATA)
    classes = list(set(tmp_df['ImagePath'].str.partition('/')[0].values.tolist()))
    arr = tmp_df['ImagePath'].values.tolist()
    for x in classes:
        temp_one_class=[]
        for im in arr:
            if im.split('/')[0]==x:
                temp_one_class.append(im)
        train_set = train_set + temp_one_class[:int(trainp*len(temp_one_class))]
        valid_set = valid_set + temp_one_class[int(trainp*len(temp_one_class)) : int(trainp*len(temp_one_class)) + int(validp*len(temp_one_class))]
        test_set = test_set + temp_one_class[int(trainp*len(temp_one_class)) + int(validp*len(temp_one_class)) : ]

    #print(classes)
    TRAIN_DATA = IMG_PATH + "filelist-train.txt"
    with open(TRAIN_DATA, "w") as f1:
        f1.write("ImagePath\n")
        for line in train_set:
            f1.write(line+"\n")

    TEST_DATA = IMG_PATH + "filelist-test.txt"
    with open(TEST_DATA, "w") as f1:
        f1.write("ImagePath\n")
        for line in test_set:
            f1.write(line+"\n")

    VALID_DATA = IMG_PATH + "filelist-valid.txt"
    with open(VALID_DATA, "w") as f1:
        f1.write("ImagePath\n")
        for line in valid_set:
            f1.write(line+"\n")

    return TRAIN_DATA,TEST_DATA,VALID_DATA
