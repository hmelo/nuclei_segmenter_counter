# nuclei_segmenter_counter

Deployment of U-net for nuclei segmentation and counting.

## Summary

The purpose of this project is to demonstrate a simple application of deep learning for solving a problem within digital pathology. Some of the main image analysis tasks in digital pathology include detection and counting (e.g., mitotic events), segmentation (e.g., nuclei), and tissue classification (e.g., cancerous vs. non-cancerous). Unfortunately, issues with slide preparation, variations in staining and scanning across sites, and vendor platforms, as well as biological variance, such as the presentation of different grades of disease, make these image analysis tasks particularly challenging [1]. Traditional approaches rely heavily on task-specific “handcrafted” features and require extensive manual tuning to accommodate these variances which is expensive and inefficient [1,2]. Deep learning may be ideally suited to tackle these challenges as it offers a more domain agnostic approach combining both feature discovery and implementation to maximally discriminate between the classes of interest.

The first step towards tackling these issues is perhaps to create a general AI tool to help pathologists segment and count cells within a sample slide.

In this notebook, I demonstrate such a cell segmentation tool by training a deep convolutional neural network on cell images acquired from the 2018 Science Bowl Kaggle competition (data available after registration). Specifically, training was performed on 670 labeled images using a U-Net architecture [1], with a 90/10 train/validation split, testing was done on an additional 65 images. This dataset is particularly useful as it includes different kinds of images with varying sizes, colours, modalities. Thus, the network learns to segment cells across different conditions without handcrafting features. The network was implemented using Keras and Tensorflow. Additionally, I created a simple function to count the number of cells after segmentation.

### The 'Data_Science_Bowl_2018_preProc_counting.ipynb' notebook provides a step-by-step demonstration with the following sections:

1. Import required libraries
2. Load datasets
3. Image preprocessing
4. Data Visualization
5. Setup U-net model
6. Training
7. Prediction
8. Visualize Results
9. Cell Counting

### Results

Overall the model does a good job at segmenting images and counting cells. Future development will focus on classification and counting of different cell types.

<img src="/images/segment_results.png" width="85%">
<img src="/images/counting.png" width="85%">

### Final thoughts
The final result from this pipeline is a segmented image with cells counted. It works well as a first prototype and proof of concept but there is much room for improvement; for example, some of the cells overlap and get clustered together as one single cell, which also leads to under counting. Future post-processing using water shed techniques should help with this. Additionally, I'd like to try other network architectures including R-CNN.

[1] Janowczyk, A., & Madabhushi, A. (2016). Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases. Journal of pathology informatics, 7.

[2] Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., ... & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. Medical image analysis, 42, 60-88.
