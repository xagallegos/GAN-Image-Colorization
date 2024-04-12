# Image Colorization with Generative Adversarial Networks
Xander Gallegos 
+ <aranxa.gallegos@iteso.mx>


This project revisits previous research that addressed image colorization through the implementation of convolutional autoencoders ([Image-Colorization](https://github.com/xagallegos/Image-Colorization)). Although these methods demonstrated some degree of ability to represent and generate color images, it was found that the results obtained did not meet the desired standards in terms of clarity and visual quality. These limitations have prompted the exploration of new methodologies, such as Generative Adversarial Networks (GANs). This neural network architecture has shown remarkable abilities in generating realistic data and capturing high-dimensional distributions, thus motivating the transition to the use of GANs in this project to address once again the challenge of grayscale image colorization.

## Results
![output-result](output-examples/Colorization_01.png)

This project is fully grounded on the paper [*"Image-to-Image Translation with Conditional Adversarial Networks"*](https://arxiv.org/pdf/1611.07004.pdf), also known as pix2pix, which proposes a general solution to various image-to-image problems, including colorization. Additionally, modifications are implemented in the generator inspired by the proposals of Moein Shariatnia in his [GitHub repository](https://github.com/moein-shariatnia/Deep-Learning/tree/main/Image%20Colorization%20Tutorial). These adaptations aim to further enhance the performance and quality of grayscale image colorization by optimizing the generator architecture within the pix2pix framework.

## Data
For model training, the Places365 dataset was chosen due to its wide diversity of environments, which is expected to have allowed the model to learn representative features of a broad range of scenarios and contributed to the model's generalization power. It's worth noting that, given the considerable size of the complete dataset, a subset of over 36k images obtained from [Kaggle](https://www.kaggle.com/datasets/pankajkumar2002/places365) was used, more aligned with the resources allocated to the project.

## Architecture
### Generator
A pre-trained ResNet-18 model is used as the base for the generator's encoder. This model has an architecture consisting mainly of convolutional layers, pooling layers (such as MaxPooling or AvgPooling), and normalization layers, all organized in blocks and with ReLU activation function. These residual blocks allow capturing complex features and facilitate training deep networks without degrading performance during the process.

The "decoder" part of the model is filled using a Dynamic U-Net because its skip connections between the encoder and decoder layers allow access to features of different spatial scales to improve accuracy in image generation, thus correcting the problems encountered in the previous implementation of this task. Together with the pre-trained ResNet-18 base as encoder, this combination provides a robust and flexible architecture for image generation, ensuring optimal performance and adaptability to various image generation scenarios.

### Discriminator
A patch discriminator is used, where the model generates an output for each patch, which consists of a square of $n \times n$ pixels, deciding individually for each patch whether it is fake or real. The architecture consists of blocks of convolutional layers with ReLU activation and batch normalization. Using this type of model for colorization seems reasonable since the local modifications required by the task are crucial. Perhaps deciding on the entire image, as a conventional discriminator does, could overlook the subtleties involved in this task.

## Loss function
The `BCEWithLogitsLoss` function is used as the loss function, which combines a sigmoid layer and binary cross-entropy into a single class, making it more numerically stable than using a simple sigmoid followed by BCE. For the generator, the goal is to minimize the probability of correct classification by the discriminator, while for the discriminator, the objective is to maximize the probability of correct classification between real and fake images.

## Evaluation metrics
Taking as reference the work of Isola et al. (2017) in 'Image-to-Image Translation with Conditional Adversarial Networks', the ultimate goal of tasks such as colorization is typically the plausibility for a human observer. Therefore, the model will be evaluated using this approach.

## Conclusions
The results obtained greatly exceeded expectations given the limitations of resources and time allocated to the project. However, some significant issues have been identified in the model. For example, when faced with images with very white areas, which are atypical in the color distribution of the dataset, the model tends to generate fills with colors that are clearly incorrect to the human eye. Nevertheless, the hypothesis is raised that these issues could be easily corrected with more training, possibly through specific targeting towards this type of images.
