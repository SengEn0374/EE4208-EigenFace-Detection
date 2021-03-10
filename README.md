# EE4208-1
## Face Detection and Recognition

### eigenvectors file
pretrained eigenfaces, "cfd__reduced_eigenvecs.npy" in google drive [link](https://drive.google.com/file/d/16HaGSCap8h1REnUy4orzchpSoFCZelFe/view?usp=sharing) 

### Requirements
-python3.x
-opencv

### Instruction
#### training
1. Sizable large dataset of faces is required for calculating princpal components. [CFD](https://chicagofaces.org/default/) face dataset used.
2. Image alignment required for all faces best results ie. eyes line up with eyes, nose lines with noses.
3. Run Eigenface.py to produce princpal component set of eigenvectors.npy file and mean_faces.npy used for training
4. Run mean_face to produce mean of each ID (john_lim_mean.npy) in separate dataset ie data is group according to IDs here. ie John_Lim > John_smile.jpg, John_frown.jpg, ...  

#### testing/prediction
-Run predict live cam py file

### Running prediction only
-Run predict_live_cam.py
