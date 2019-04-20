# EENA
The project is the post-training of the neural architectures designed by EENA (Efficient Evolution of Neural Architectures).  
If you want to execute this project, you can run the Posttraining.py with the default configurations.   
Serveral configurations:  
  \>Standard augmentation: We normalize the images using channel means and standard deviations for preprocessing and 
  apply a standard data augmentation scheme (zero-padding with 4 pixels on each side to obtain a 40\*40 pixels image,
then randomly cropping it to size 32\*32 and randomly flipping the image horizontally).   
  \>Cutout: n_holes = 1, length = 16;  
  \>Mixup: α = 1.  
The model is trained on the full training dataset until convergence using SGDR with a batch size of 128.  
The final test error on CIFAR-10 can achieve around 2.60.  
