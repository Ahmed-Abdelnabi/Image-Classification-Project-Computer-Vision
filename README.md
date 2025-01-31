# Image-Classification-Project-Computer-Vision

This project focuses on image classification using Convolutional Neural Networks (CNNs) and transfer learning techniques to predict images across six classes: (Mountain, Buildings, Street, Sea, Glacier, and Forest).

A custom CNN model was developed (called ‘My-Net’), and also transfer learning was applied using pre-trained architectures such as VGG16, VGG19, InceptionV3, ResNet152V, and DenseNet201. 
Two voting ensembles and two average ensembles were constructed to improve performance. 

The best-performing model was an average ensemble combining My-Net, Res-Net, and Dense-Net, achieving 94% accuracy, 94% precision, 94% recall, and 94% F1-score on the test dataset. 

Future work could explore additional data augmentation techniques, hyperparameter tuning, and testing other pre-trained models like EfficientNet or MobileNet.
