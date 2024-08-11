<h1> University of London Final Project: L.E.G.O Net - A hardware-restricted approach to training Melanoma classifiers </h1>


<h2> Abstract </h2>

<p>Melanona is one of the deadliest types of cancer if not detected earlier. The need for systems that support/enhance early detection is growing daily as cases follow an upward trajectory. However, due to privacy concerns and high annotation costs associated with medical images, available and well-annotated medical data is extremely scarce. This project aims to solve the hardware constraint, the class imbalance, and the lack of labeled data by taking a simple approach: subdivide the available annotated material into smaller patches and then use those patches to train a Conditional Adversarial Network to generate new synthetic patches. Then take those patches, train a CNN to perform binary classification on the subdivided components, and teach it to evaluate patch groups composing an image, outputting a patch-level probability to pool it into the final classification. </p>



<h2> Repository Structure </h2>

<p>This repository is divided into two main blocks: a research block and a production block. The research block contains all iterations, files, and a rather wild and unorganized carry of attempts to find the best combinations of architectures, hyperparameters, backbones, and information to perform the aforementioned task. The production block contains the best-performing setup for the experiment on the two main datasets used: The ISIC and HAM Datasets.</p>
