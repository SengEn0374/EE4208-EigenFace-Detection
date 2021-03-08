import os

file = './cfd_crop.lst'
f = open(file, 'w')

folder = './cfd_crop'
imgs = os.listdir(folder)

for i, img in enumerate(imgs):
	img_dir = os.path.join(folder, img)
	if i == len(imgs)-1:
		f.write(img_dir)
	else:
		f.write(img_dir + ',')
f.close()