<h1 style="text-align: center; color: #2E86C1;">University of London Final Project: L.E.G.O Net - Locally Enhanced Observations Network. A Hardware-Restricted Approach to Training Melanoma Classifiers</h1>

<h2 style="color: #2874A6;">Abstract</h2>

<p>Melanoma is one of the deadliest types of cancer if not detected early. The need for systems that support and enhance early detection is growing daily as cases follow an upward trajectory. However, due to privacy concerns and high annotation costs associated with medical images, available and well-annotated medical data is extremely scarce. This project aims to address hardware constraints, class imbalance, and the lack of labeled data by employing a simple yet effective approach:</p>

<ul style="line-height: 1.6;">
  <li>Subdivide the available annotated material into smaller patches.</li>
  <li>Use these patches to train a Conditional Adversarial Network to generate new synthetic patches.</li>
  <li>Train a CNN to perform binary classification on the subdivided components.</li>
  <li>Teach the CNN to evaluate patch groups composing an image, outputting a patch-level probability that can be pooled into the final classification.</li>
</ul>

<h2 style="color: #2874A6;">Repository Structure</h2>

<p>This repository is divided into two main blocks:</p>

<ol style="line-height: 1.6;">
  <li>
    <strong>Research Block:</strong> This contains most of the iterations, files, and a rather unorganized trove of attempts to find the best combinations of architectures, hyperparameters, backbones, and information to accomplish the project's goals.
  </li>
  <li>
    <strong>Production Pipeline Block:</strong> This contains the best-performing setup for the experiment on the two main datasets used: The ISIC2020 and HAM10000 datasets.
  </li>
</ol>

<p>Please note that due to the public nature of this repository, the trained model files are not uploaded because of their large size. Instead, "dummy files" have been placed in the repository. These serve to validate the paths in the repository and demonstrate how the script in the pipeline pulls the different pre-trained models into the ensemble.</p>

<h2 style="color: #2874A6;">Overview</h2>

<p>L.E.G.O. Net is a deep learning architecture designed to address the challenges of data scarcity and imbalance in medical image classification, particularly in melanoma detection. The system leverages the power of Conditional Generative Adversarial Networks (cGANs), Vision Transformers (CvT), and Convolutional Neural Networks (CNNs) to create a robust pipeline that improves diagnostic accuracy by focusing on sub-regions of images, which are treated as discrete, non-overlapping patches. The current validation accuracy of the model stands at 91%.</p>

<h2 style="color: #2874A6;">Key Features</h2>

<ul style="line-height: 1.6;">
  <li><strong>Synthetic Data Generation:</strong> The cGAN component generates realistic sub-regions of melanoma images to augment the training dataset, addressing the challenge of class imbalance.</li>
  <li><strong>Vision Transformer:</strong> The CvT model learns spatial and contextual relationships between patches, improving the model's ability to generalize across different melanoma images.</li>
  <li><strong>Convolutional Neural Network:</strong> The CNN performs final classification by evaluating the aggregated features from the CvT, providing a binary output (malignant/benign).</li>
  <li><strong>Data Privacy:</strong> The model is designed to run locally, ensuring that sensitive medical data remains secure.</li>
  <li><strong>Interpretability:</strong> The use of patch-based analysis and saliency maps enhances the interpretability of the model’s decisions.</li>
</ul>

<h2 style="color: #2874A6;">Installation</h2>

<h3 style="color: #1F618D;">Prerequisites</h3>

<p>Ensure you have the following installed:</p>

<ul style="line-height: 1.6;">
  <li>Python 3.x</li>
  <li>TensorFlow</li>
  <li>OpenCV</li>
  <li>Albumentations</li>
  <li>NumPy</li>
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Scikit-learn</li>
</ul>

<p>Install the required Python packages using pip:</p>

<pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;"><code>pip install tensorflow opencv-python-headless albumentations numpy pandas matplotlib scikit-learn</code></pre>

<h3 style="color: #1F618D;">Directory Structure</h3>

<p>Place the required models and scripts in the correct directories as specified in the configuration file (<code>config.json</code>). The directory structure should look like this:</p>

<pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
<code>
L.E.G.O_Net/
│
├── data/
│   ├── images/
│   ├── patches/
│   └── config.json
│
├── models/
│   ├── cGAN.h5
│   ├── CvT.h5
│   └── CNN.h5
│
├── scripts/
│   ├── image_cropper.py
│   ├── patch_generator.py
│   ├── patch_organiser.py
│   └── pipeline.py
│
├── notebooks/
│   ├── cGAN.ipynb
│   ├── CvT.ipynb
│   └── CNN.ipynb
│
└── README.md
</code>
</pre>

<h2 style="color: #2874A6;">Configuration</h2>

<p>Edit the <code>config.json</code> file to specify the paths and parameters for the pipeline:</p>

<pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;"><code>
{
    "paths": {
        "input_image": "data/images/input_image.jpg",
        "pretrained_cgan_model": "models/cGAN.h5",
        "pretrained_cvt_model": "models/CvT.h5",
        "pretrained_cnn_model": "models/CNN.h5",
        "csv_path": "data/images/metadata.csv",
        "img_dir": "data/images"
    },
    "data_augmentation": {
        "use_gaussian_noise": true,
        "use_motion_blur": true,
        "use_grid_distortion": false,
        "use_sharpen": true,
        "use_emboss": false,
        "use_random_brightness_contrast": true,
        "width_shift_range": 0.1,
        "zoom_range": 0.2,
        "rotation_range": 30,
        "clahe_clip_limit": 2.0,
        "brightness_range": [0.8, 1.2]
    }
}
</code></pre>

<h2 style="color: #2874A6;">Usage</h2>

<h3 style="color: #1F618D;">Running the Pipeline</h3>

<p>Execute the pipeline to classify an image as malignant or benign:</p>

<pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;"><code>python lego_net/lego_net.py</code></pre>

<p>The pipeline follows these steps:</p>

<ol style="line-height: 1.6;">
  <li><strong>Preprocessing:</strong> Crops and centers the lesion in the image using <code>image_cropper.py</code>.</li>
  <li><strong>Patch Generation:</strong> Divides the preprocessed image into patches using <code>patch_generator.py</code>.</li>
  <li><strong>Patch Augmentation:</strong> Augments the patches using the cGAN model.</li>
  <li><strong>Feature Extraction:</strong> Extracts features from the patches using the CvT model.</li>
  <li><strong>Classification:</strong> Classifies the patches using the CNN model.</li>
  <li><strong>Aggregation:</strong> Aggregates patch-level classifications to produce a final image-level classification.</li>
</ol>

<h3 style="color: #1F618D;">Model Training</h3>

<p>You can train the models independently using the provided Jupyter notebooks:</p>

<ul style="line-height: 1.6;">
  <li><strong>cGAN:</strong> <code>notebooks/cGAN.ipynb</code></li>
  <li><strong>CvT:</strong> <code>notebooks/CvT.ipynb</code></li>
  <li><strong>CNN:</strong> <code>notebooks/CNN.ipynb</code></li>
</ul>

<p>Each notebook contains the code to train and validate the respective model.</p>

<h2 style="color: #2874A6;">Evaluation</h2>

<p>L.E.G.O. Net achieves a validation accuracy of 91%. The performance metrics for different components and the full ensemble model are detailed in the final report. The system’s effectiveness is demonstrated through its ability to classify challenging melanoma cases with a high degree of accuracy.</p>


<h2 style="color: #2874A6;">Acknowledgments</h2>

<p>Special thanks to the University of London and all the contributors to the open-source libraries used in this project. Their work has been instrumental in the development of L.E.G.O. Net.</p>
