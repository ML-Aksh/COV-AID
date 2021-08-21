# COV-AID

# PROJECT SUMMARY


# PAPER ABSTRACT
With COVID-19 cases rising rapidly, deep learning has emerged as a promising diagnosis technique. However, identifying the most accurate models to characterize COVID-19 patients is challenging because comparing results obtained with different types of data and acquisition processes is non-trivial. 

In this project we designed, evaluated, and compared the performance of 20 convolutional neural networks in classifying patients as COVID-19 positive, healthy, or suffering from other pulmonary lung infections based on Chest CT scans, serving as the first to consider the EfficientNet family for COVID-19 diagnosis and employ intermediate activation maps for visualizing model performance. 

All models are trained and evaluated in Python using 4173 Chest CT images from the dataset entitled “A COVID multiclass dataset of CT scans,” with 2168, 758, and 1247 images of patients that are COVID-19 positive, healthy, or suffering from other pulmonary infections, respectively. 

EfficientNet-B5 was identified as the best model with an F1 score of 0.9769±0.0046, accuracy of 0.9759±0.0048, sensitivity of 0.9788±0.0055, specificity of 0.9730±0.0057, and precision of 0.9751± 0.0051.  On an alternate 2-class dataset, EfficientNetB5 obtained an accuracy of 0.9845±0.0109, F1 score of 0.9599±0.0251, sensitivity of 0.9682±0.0099, specificity of 0.9883±0.0150, and precision of 0.9526 ± 0.0523. Intermediate activation maps and Gradient-weighted Class Activation Mappings offered human-interpretable evidence of the model’s perception of ground-class opacities and consolidations, hinting towards a promising use-case of artificial intelligence-assisted radiology tools. With a prediction speed of under 0.1 seconds on GPUs and 0.5 seconds on CPUs, our proposed model offers a rapid, scalable, and accurate diagnostic for COVID-19. 

# PREPRINT
Check out our paper "Efficient and Visualizable Convolutional Neural Networks for COVID-19 Classification Using Chest CT: at: https://arxiv.org/abs/2012.11860

# DATASET
We used two datasets for our study. Dataset 1 was used for extensive model training, hyperparameter tuning, and evaluation. Dataset 2 was included for comparitive reasons and evaluating the effectiveness of our proposed method on new, alternative datasets. 
<ol>
  <li> <a href="https://www.kaggle.com/plameneduardo/a-covid-multiclass-dataset-of-ct-scans"> Dataset 1: A COVID Multiclass Dataset of CT Scans: </a>  4173 CT images of 210 different patients hospitalized in Sao Paulo Brazil. 2168 images of 80 patients infected with COVID-19, 758 images of 50 healthy patients, and 1247 images of 80 patients with other pulmonary infections (~20 images/person). All images were grayscale in nature, collected from patients in Sao Paulo, Brazil, and made freely accessible through Kaggle by Soares E. et al 
  <li> <a href="https://github.com/mr7495/COVID-CTset"> Dataset 2: COVID-CTset </a> Contains 63849 CT images from 377 patients (96 COVID-19 positive and 283 Covid-19 negative). To facilitate faster testing, we considered a subset of the data comprised of 12058 images from those 377 patients.
</ol>
