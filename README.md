# -NEURAL-STYLE-TRANSFER

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*:  NALLAGATLA ADITHYA

*INTERN ID*: CT04DM719

*DOMAIN*: Artificial intelligence

*DURATION*:  4 weeks

*MENTOR*:  Neela Santhosh

#discription:

This Python script performs neural style transfer—a technique in computer vision and deep learning that blends the artistic style of one image with the content of another using a pretrained convolutional neural network (CNN). The project is executed on Google Colab, a cloud-based platform ideal for running computationally intensive deep learning tasks without the need for local hardware or installations. Colab provides a convenient workspace with GPU support and quick access to packages like PyTorch and Torchvision, both of which are extensively used in this script.

The key libraries and tools involved are as follows:

PyTorch (torch): A popular open-source deep learning framework used for building, training, and deploying neural networks. It handles tensor operations, GPU management, loss computations, and model optimization.

Torchvision (torchvision.models): Provides pretrained models like VGG19, which are used in this project as the feature extractor.

PIL (Image, ImageFile): The Python Imaging Library is used for image loading, conversion, and preprocessing.

Matplotlib (pyplot): Used to visually display the final stylized output in the notebook.

Google Colab: A cloud-based Jupyter notebook environment that provides free access to GPUs and preinstalled packages, making it perfect for this deep learning task.

The process begins with loading two images: a content image (smile.jpg) that provides the structure or layout and a style image (styles.jpg) that provides artistic texture and color patterns. These images are resized and transformed into tensors for processing by the neural network.

A pretrained VGG19 CNN model is loaded and used in feature extraction mode. Specific layers of this network are chosen to represent content features (such as spatial structures) and style features (such as color textures and brush strokes). Feature extraction is achieved by feeding the images through the model and capturing intermediate outputs.

To synthesize the final image, the script clones the content image and enables gradient optimization to iteratively modify it. The objective is to minimize a loss function that combines two key components:

Content Loss: Measures how different the generated image is from the original content image in terms of structure.

Style Loss: Compares the style representation (using Gram matrices) of the generated image to that of the style image.

Each iteration updates the target image by calculating gradients of the combined loss, and the Adam optimizer adjusts pixel values accordingly. After 100 iterations, the final stylized image is displayed using matplotlib and saved as stylized_output.jpg.

This approach has wide-ranging applications in:

Digital Art & Design: Artists and content creators can generate unique stylized artwork.

Augmented Reality & Filters: Real-time implementation in apps for stylistic transformation of video or camera input.

Game Development: Applying consistent stylistic themes across game assets.

Education & Research: Demonstrating core principles of convolutional networks, optimization, and feature learning.

By running this in Colab, even users without powerful local GPUs can explore the creative potential of AI through an interactive, easy-to-use interface

#OUTPUT

![Image](https://github.com/user-attachments/assets/e1a4aaf3-2d2a-4bac-8e2b-5375adffe0c7)
