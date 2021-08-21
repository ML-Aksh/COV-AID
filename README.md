# COV-AID

# PROJECT SUMMARY 
_Quick Summary of what I did and how you can use these tools_
<ul>
  <li> Designed, evaluated, and compared the performance of 20 convolutional neural networks in classifying CT images as COVDI-19 positive, healthy, and as suffering from other pulmonary lung infections
  <li> Established efficacy of using EfficientNets for COVID-19 Diagnosis
  <li> Employed Intermediate Activation Maps and Gradient-weighted class activation maps (GradCAMs) to visualize model performance.
  <li> <b> Provide an easily adaptable and adjustable pipeline for image recognition tasks with 20 models including EfficientNets, ResNets, DenseNets, InceptionNets, Xception, VGGs with a <u> single line of code. </u> Customizability features include: </b>
      <ol>
        <li> Base CNN Architecture to use. Currently supported options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, ResNet50, ResNet50V2, ResNet101v2, ResNet152v2, InceptionResNetv2, VGG16, VGG19, InceptionResNetv2, DenseNet121, DenseNet169, DenseNet201, Xception
        <li> Number of training epochs
        <li> GPU vs. CPU Based Training
        <li> Callbacks: 3 Options supported currently: (0) Full Call Backs, (1) No TensorBoard, (2) No Callbacks
        <li> Image Shape to transform input images before inputting them to the CNN Pipeline
        <li> Toggle Batch Normalization off/on
        <li> Toggle Dropout on or off & Dropout Weights
        <li> Toggle transfer learning on or off
        <li> Choice between standard architecture and standard + additional classification layers
      </ol>
</ul>

# NAVIGATING THIS REPO


# DATASETS
We used two datasets for our study. Dataset 1 was used for extensive model training, hyperparameter tuning, and evaluation. Dataset 2 was included for comparitive reasons and evaluating the effectiveness of our proposed method on new, alternative datasets.
<ol>
  <li> <a href="https://www.kaggle.com/plameneduardo/a-covid-multiclass-dataset-of-ct-scans"> Dataset 1: A COVID Multiclass Dataset of CT Scans: </a>  4173 CT images of 210 different patients hospitalized in Sao Paulo Brazil. 2168 images of 80 patients infected with COVID-19, 758 images of 50 healthy patients, and 1247 images of 80 patients with other pulmonary infections (~20 images/person). All images were grayscale in nature, collected from patients in Sao Paulo, Brazil, and made freely accessible through Kaggle by Soares E. et al 
    <ul>
      <li> <b> We renormalized image names for easier use within the scripts and have uploaded the revised version of the dataset within the repository. </b>
    </ul>
  <li> <a href="https://github.com/mr7495/COVID-CTset"> Dataset 2: COVID-CTset </a> Contains 63849 CT images from 377 patients (96 COVID-19 positive and 283 Covid-19 negative). To facilitate faster testing, we considered a subset of the data comprised of 12058 images from those 377 patients.
</ol>


# PREPRINT
Check out our paper "Efficient and Visualizable Convolutional Neural Networks for COVID-19 Classification Using Chest CT": at: https://arxiv.org/abs/2012.11860


# PAPER ABSTRACT
With COVID-19 cases rising rapidly, deep learning has emerged as a promising diagnosis technique. However, identifying the most accurate models to characterize COVID-19 patients is challenging because comparing results obtained with different types of data and acquisition processes is non-trivial. 

In this project we designed, evaluated, and compared the performance of 20 convolutional neural networks in classifying patients as COVID-19 positive, healthy, or suffering from other pulmonary lung infections based on Chest CT scans, serving as the first to consider the EfficientNet family for COVID-19 diagnosis and employ intermediate activation maps for visualizing model performance. 

All models are trained and evaluated in Python using 4173 Chest CT images from the dataset entitled “A COVID multiclass dataset of CT scans,” with 2168, 758, and 1247 images of patients that are COVID-19 positive, healthy, or suffering from other pulmonary infections, respectively. 

EfficientNet-B5 was identified as the best model with an F1 score of 0.9769±0.0046, accuracy of 0.9759±0.0048, sensitivity of 0.9788±0.0055, specificity of 0.9730±0.0057, and precision of 0.9751± 0.0051.  On an alternate 2-class dataset, EfficientNetB5 obtained an accuracy of 0.9845±0.0109, F1 score of 0.9599±0.0251, sensitivity of 0.9682±0.0099, specificity of 0.9883±0.0150, and precision of 0.9526 ± 0.0523. Intermediate activation maps and Gradient-weighted Class Activation Mappings offered human-interpretable evidence of the model’s perception of ground-class opacities and consolidations, hinting towards a promising use-case of artificial intelligence-assisted radiology tools. With a prediction speed of under 0.1 seconds on GPUs and 0.5 seconds on CPUs, our proposed model offers a rapid, scalable, and accurate diagnostic for COVID-19. 

# RESULTS

**COVID-19 Positive Images**

| #  | Model             | F1              | Accuracy        | Sensitivity     | Specificity     | Precision       |
|----|-------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1  | DenseNet121       | 0.9709 ± 0.0059 | 0.9699 ± 0.0061 | 0.9655 ± 0.0103 | 0.9747 ± 0.0054 | 0.9767 ± 0.0047 |
| 2  | DenseNet169       | 0.9729 ± 0.0065 | 0.9719 ± 0.0067 | 0.9683 ± 0.0101 | 0.9759 ± 0.0072 | 0.9779 ± 0.0063 |
| 3  | DenseNet201       | 0.9733 ± 0.0058 | 0.9723 ± 0.0061 | 0.9703 ± 0.0090 | 0.9743 ± 0.0063 | 0.9766 ± 0.0054 |
| 4  | EfficientNetB0    | 0.9648 ± 0.0051 | 0.9633 ± 0.0054 | 0.9658 ± 0.0083 | 0.9608 ± 0.0080 | 0.9644 ± 0.0069 |
| 5  | EfficientNetB1    | 0.9300 ± 0.0250 | 0.9276 ± 0.0244 | 0.9323 ± 0.0298 | 0.9226 ± 0.0361 | 0.9350 ± 0.0260 |
| 6  | EfficientNetB2    | 0.9546 ± 0.0062 | 0.9530 ± 0.0066 | 0.9476 ± 0.0090 | 0.9590 ± 0.0080 | 0.9622 ± 0.0070 |
| 7  | EfficientNetB3    | 0.9594 ± 0.0065 | 0.9580 ± 0.0066 | 0.9552 ± 0.0106 | 0.9613 ± 0.0070 | 0.9642 ± 0.0064 |
| 8  | EfficientNetB4    | 0.9647 ± 0.0072 | 0.9634 ± 0.0074 | 0.9637 ± 0.0113 | 0.9635 ± 0.0074 | 0.9663 ± 0.0068 |
| 9  | EfficientNetB5    | 0.9769 ± 0.0046 | 0.9759 ± 0.0048 | 0.9788 ± 0.0055 | 0.9730 ± 0.0057 | 0.9751 ± 0.0051 |
| 10 | EfficientNetB6    | 0.9614 ± 0.0053 | 0.9597 ± 0.0056 | 0.9661 ± 0.0080 | 0.9532 ± 0.0088 | 0.9573 ± 0.0078 |
| 11 | EfficientNetB7    | 0.9448 ± 0.0074 | 0.9432 ± 0.0077 | 0.9397 ± 0.0131 | 0.9475 ± 0.0087 | 0.9511 ± 0.0077 |
| 12 | InceptionResNetV2 | 0.9450 ± 0.0069 | 0.9427 ± 0.0074 | 0.9464 ± 0.0124 | 0.9392 ± 0.0097 | 0.9443 ± 0.0083 |
| 13 | InceptionV3       | 0.9567 ± 0.0070 | 0.9549 ± 0.0072 | 0.9587 ± 0.0117 | 0.9509 ± 0.0099 | 0.9554 ± 0.0087 |
| 14 | ResNet101V2       | 0.9383 ± 0.0107 | 0.9364 ± 0.0116 | 0.9289 ± 0.0156 | 0.9450 ± 0.0151 | 0.9490 ± 0.0128 |
| 15 | ResNet152V2       | 0.9407 ± 0.0099 | 0.9380 ± 0.0107 | 0.9441 ± 0.0158 | 0.9315 ± 0.0185 | 0.9389 ± 0.0139 |
| 16 | ResNet50          | 0.9638 ± 0.0061 | 0.9625 ± 0.0062 | 0.9609 ± 0.0104 | 0.9643 ± 0.0093 | 0.9672 ± 0.0084 |
| 17 | ResNet50V2        | 0.9335 ± 0.0092 | 0.9308 ± 0.0099 | 0.9328 ± 0.0154 | 0.9292 ± 0.0185 | 0.9361 ± 0.0143 |
| 18 | VGG16             | 0.8932 ± 0.0107 | 0.8889 ± 0.0111 | 0.8954 ± 0.0190 | 0.8828 ± 0.0166 | 0.8930 ± 0.0136 |
| 19 | VGG19             | 0.8673 ± 0.0189 | 0.8558 ± 0.0304 | 0.8838 ± 0.0219 | 0.8247 ± 0.0748 | 0.8599 ± 0.0337 |
| 20 | Xception          | 0.9491 ± 0.0062 | 0.9470 ± 0.0064 | 0.9510 ± 0.0118 | 0.9432 ± 0.0112 | 0.9482 ± 0.0096 |

**Healthy Images**
|    | Model             | F1              | Accuracy        | Sensitivity     | Specificity     | Precision       |
|----|-------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1  | DenseNet121       | 0.7835 ± 0.0263 | 0.9193 ± 0.0099 | 0.8043 ± 0.0334 | 0.9445 ± 0.0102 | 0.7700 ± 0.0350 |
| 2  | DenseNet169       | 0.7835 ± 0.0277 | 0.9195 ± 0.0102 | 0.8045 ± 0.0365 | 0.9448 ± 0.0100 | 0.7704 ± 0.0330 |
| 3  | DenseNet201       | 0.7854 ± 0.0281 | 0.9197 ± 0.0110 | 0.8055 ± 0.0302 | 0.9449 ± 0.0111 | 0.7723 ± 0.0385 |
| 4  | EfficientNetB0    | 0.7909 ± 0.0247 | 0.9201 ± 0.0099 | 0.8259 ± 0.0271 | 0.9410 ± 0.0095 | 0.7651 ± 0.0321 |
| 5  | EfficientNetB1    | 0.7307 ± 0.0492 | 0.8965 ± 0.0176 | 0.7875 ± 0.0554 | 0.9207 ± 0.0183 | 0.7041 ± 0.0445 |
| 6  | EfficientNetB2    | 0.7949 ± 0.0241 | 0.9197 ± 0.0099 | 0.8481 ± 0.0274 | 0.9357 ± 0.0102 | 0.7551 ± 0.0315 |
| 7  | EfficientNetB3    | 0.7912 ± 0.0233 | 0.9194 ± 0.0095 | 0.8332 ± 0.0268 | 0.9387 ± 0.0098 | 0.7606 ± 0.0321 |
| 8  | EfficientNetB4    | 0.7925 ± 0.0276 | 0.9220 ± 0.0106 | 0.8193 ± 0.0335 | 0.9448 ± 0.0097 | 0.7744 ± 0.0322 |
| 9  | EfficientNetB5    | 0.8217 ± 0.0249 | 0.9322 ± 0.0109 | 0.8488 ± 0.0185 | 0.9504 ± 0.0118 | 0.8010 ± 0.0383 |
| 10 | EfficientNetB6    | 0.7891 ± 0.0242 | 0.9177 ± 0.0099 | 0.8465 ± 0.0291 | 0.9337 ± 0.0093 | 0.7448 ± 0.0301 |
| 11 | EfficientNetB7    | 0.7810 ± 0.0243 | 0.9161 ± 0.0100 | 0.8169 ± 0.0265 | 0.9383 ± 0.0096 | 0.7538 ± 0.0315 |
| 12 | InceptionResNetV2 | 0.7727 ± 0.0267 | 0.9154 ± 0.0102 | 0.7918 ± 0.0370 | 0.9429 ± 0.0104 | 0.7622 ± 0.0360 |
| 13 | InceptionV3       | 0.7673 ± 0.0295 | 0.9130 ± 0.0112 | 0.7912 ± 0.0405 | 0.9402 ± 0.0115 | 0.7541 ± 0.0382 |
| 14 | ResNet101V2       | 0.7408 ± 0.0347 | 0.9035 ± 0.0116 | 0.7677 ± 0.0448 | 0.9333 ± 0.0108 | 0.7234 ± 0.0364 |
| 15 | ResNet152V2       | 0.7596 ± 0.0329 | 0.9093 ± 0.0120 | 0.7931 ± 0.0423 | 0.9351 ± 0.0115 | 0.7367 ± 0.0384 |
| 16 | ResNet50          | 0.7784 ± 0.0253 | 0.9174 ± 0.0101 | 0.7961 ± 0.0290 | 0.9442 ± 0.0099 | 0.7670 ± 0.0343 |
| 17 | ResNet50V2        | 0.7548 ± 0.0283 | 0.9081 ± 0.0106 | 0.7824 ± 0.0395 | 0.9358 ± 0.0112 | 0.7379 ± 0.0346 |
| 18 | VGG16             | 0.7414 ± 0.0226 | 0.8986 ± 0.0100 | 0.7984 ± 0.0288 | 0.9208 ± 0.0102 | 0.6957 ± 0.0280 |
| 19 | VGG19             | 0.6639 ± 0.0309 | 0.8749 ± 0.0120 | 0.6499 ± 0.0731 | 0.9248 ± 0.0136 | 0.6623 ± 0.0341 |
| 20 | Xception          | 0.7806 ± 0.0291 | 0.9179 ± 0.0113 | 0.8013 ± 0.0354 | 0.9436 ± 0.0109 | 0.7684 ± 0.0377 |

**Other Pulmonary Infections Images**
| #  | Model             | F1              | Accuracy        | Sensitivity     | Specificity     | Precision       |
|----|-------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1  | DenseNet121       | 0.8239 ± 0.0242 | 0.8966 ± 0.0131 | 0.8188 ± 0.0313 | 0.9293 ± 0.0100 | 0.8315 ± 0.0234 |
| 2  | DenseNet169       | 0.8245 ± 0.0244 | 0.8971 ± 0.0128 | 0.8188 ± 0.0313 | 0.9301 ± 0.0108 | 0.8333 ± 0.0243 |
| 3  | DenseNet201       | 0.8262 ± 0.0256 | 0.8989 ± 0.0133 | 0.8178 ± 0.0344 | 0.9325 ± 0.0091 | 0.8374 ± 0.0213 |
| 4  | EfficientNetB0    | 0.8207 ± 0.0249 | 0.8980 ± 0.0129 | 0.7976 ± 0.0300 | 0.9401 ± 0.0088 | 0.8483 ± 0.0233 |
| 5  | EfficientNetB1    | 0.7482 ± 0.0535 | 0.8623 ± 0.0244 | 0.7210 ± 0.0571 | 0.9221 ± 0.0180 | 0.7889 ± 0.0463 |
| 6  | EfficientNetB2    | 0.8121 ± 0.0246 | 0.8931 ± 0.0127 | 0.7901 ± 0.0311 | 0.9363 ± 0.0095 | 0.8398 ± 0.0233 |
| 7  | EfficientNetB3    | 0.8170 ± 0.0243 | 0.8952 ± 0.0129 | 0.7979 ± 0.0310 | 0.9360 ± 0.0099 | 0.8414 ± 0.0234 |
| 8  | EfficientNetB4    | 0.8288 ± 0.0247 | 0.9009 ± 0.0138 | 0.8143 ± 0.0301 | 0.9373 ± 0.0131 | 0.8496 ± 0.0280 |
| 9  | EfficientNetB5    | 0.8385 ± 0.0278 | 0.9077 ± 0.0140 | 0.8172 ± 0.0367 | 0.9458 ± 0.0084 | 0.8643 ± 0.0225 |
| 10 | EfficientNetB6    | 0.8157 ± 0.0200 | 0.8963 ± 0.0103 | 0.7747 ± 0.0273 | 0.9483 ± 0.0064 | 0.8648 ± 0.0166 |
| 11 | EfficientNetB7    | 0.8038 ± 0.0210 | 0.8856 ± 0.0111 | 0.7905 ± 0.0277 | 0.9262 ± 0.0106 | 0.8235 ± 0.0235 |
| 12 | InceptionResNetV2 | 0.7919 ± 0.0239 | 0.8790 ± 0.0123 | 0.7798 ± 0.0303 | 0.9210 ± 0.0099 | 0.8073 ± 0.0245 |
| 13 | InceptionV3       | 0.7963 ± 0.0286 | 0.8824 ± 0.0150 | 0.7799 ± 0.0367 | 0.9254 ± 0.0123 | 0.8177 ± 0.0275 |
| 14 | ResNet101V2       | 0.7837 ± 0.0279 | 0.8717 ± 0.0163 | 0.7818 ± 0.0311 | 0.9101 ± 0.0163 | 0.7900 ± 0.0340 |
| 15 | ResNet152V2       | 0.7835 ± 0.0254 | 0.8766 ± 0.0123 | 0.7600 ± 0.0381 | 0.9258 ± 0.0115 | 0.8154 ± 0.0244 |
| 16 | ResNet50          | 0.8177 ± 0.0241 | 0.8933 ± 0.0128 | 0.8114 ± 0.0323 | 0.9279 ± 0.0096 | 0.8272 ± 0.0232 |
| 17 | ResNet50V2        | 0.7697 ± 0.0275 | 0.8668 ± 0.0138 | 0.7552 ± 0.0355 | 0.9137 ± 0.0118 | 0.7888 ± 0.0273 |
| 18 | VGG16             | 0.6865 ± 0.0299 | 0.8240 ± 0.0136 | 0.6544 ± 0.0392 | 0.8959 ± 0.0146 | 0.7304 ± 0.0305 |
| 19 | VGG19             | 0.6346 ± 0.0303 | 0.7840 ± 0.0162 | 0.6011 ± 0.0692 | 0.8613 ± 0.0208 | 0.6535 ± 0.0284 |
| 20 | Xception          | 0.8017 ± 0.0299 | 0.8854 ± 0.0159 | 0.7863 ± 0.0360 | 0.9268 ± 0.0122 | 0.8210 ± 0.0289 |
