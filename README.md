# Reproducibility study: Unsupervised Learning for Cell-Level Visual Representation in Histopathology Images With Generative Adversarial Networks

This repository contains the implementation of our reproducibility study of the paper *Unsupervised Learning for Cell-level Visual Representation in Histopathology Images with Generative Adversarial Networks*, 
Bo Hu♯ , Ye Tang♯ , Eric I-Chao Chang, Yubo Fan, Maode Lai and Yan Xu*  (* corresponding author; ♯ equal contribution), [arxiv](https://arxiv.org/abs/1711.11317), [IEEE](https://ieeexplore.ieee.org/document/8402089)

Original paper's code can be found on [*nu-gan*](https://github.com/bohu615/nu_gan)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To download datasets: 

* Extract all data to desired path: [Datasets](https://queensuca-my.sharepoint.com/:f:/g/personal/19nyk1_queensu_ca/EuoCEg-JUB9AuqkHWoh8WLQBDWGf3cnJLKzBJGsTkLpqsw?e=inwedL)

## Training

To train the model(s) in the paper, separate tasks can be chosen using flags.

### Unsupervised Cell-level Clustering on labeled dataset:
```train
python /path_to_utils/nu_gan.py --task 'cell_representation'
```

* To change the number of clusters:
```train
python /path_to_utils/nu_gan.py --task 'cell_representation' --dis_category 4
```
The default number of clusters (`dis_category` is 5). 

### Unsupervised Cell-level Clustering on unlabeled dataset:
```train
python /path_to_utils/nu_gan.py --task 'cell_representation_unlabeled'
```

* To change the number of clusters:
```train
python /path_to_utils/nu_gan.py --task 'cell_representation_unlabeled' --dis_category 4
```
The default number of clusters (`dis_category` is 5). 

### Cell Classification:
```train
python /path_to_utils/nu_gan.py --task 'cell_classification' --experiment_id 123456
```

`experiment_id` identifies the experiment from cell-level clustering to perform cell classification. Can be retrieved from the output folder name from `cell_representation` or `cell_representation_unlabeled`.

Other hyperparameters that don't have specified flags can be changed in `nu_gan.py`.

## Evaluation

To evaluate my model, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
