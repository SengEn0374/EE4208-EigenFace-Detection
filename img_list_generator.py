import os

# fp = './frontal_face_aligned.lst'
fp = './56x56.lst'
f = open(fp, 'w')

# dataset_path = './frontal_face_aligned'
dataset_path = './56x56'
IDs = os.listdir(dataset_path)
flg = 0
for i, ID in enumerate(IDs):
    if i == len(IDs)-1:
        flg = 1
    ID_path = os.path.join(dataset_path, ID)
    ID_imgs = os.listdir(ID_path)
    for i, img in enumerate(ID_imgs):
        img_path = os.path.join(ID_path, img)
        # print(img_path)
        if i == len(ID_imgs)-1 and flg == 1:
            f.write(img_path)
        else:
            f.write(img_path + ',')

f.close()

# check
f = open('frontal_face_aligned.lst', "r")
line = f.read()
dirs = line.split(',')
for dir in dirs:
    print(dir)


