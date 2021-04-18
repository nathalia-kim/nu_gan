# Reproducibility study: Unsupervised Learning for Cell-Level Visual Representation in Histopathology Images With Generative Adversarial Networks

This repository contains the implementation of our reproducibility study of the paper *Unsupervised Learning for Cell-level Visual Representation in Histopathology Images with Generative Adversarial Networks*, 
Bo Hu♯ , Ye Tang♯ , Eric I-Chao Chang, Yubo Fan, Maode Lai and Yan Xu*  (* corresponding author; ♯ equal contribution), [arxiv](https://arxiv.org/abs/1711.11317), [IEEE](https://ieeexplore.ieee.org/document/8402089)

Original paper's code can be found on [*nu-gan*](https://github.com/bohu615/nu_gan)

## Requirements

To install requirements:

```bash
conda create -n myenv python=3.8.5
conda activate myenv

conda install -c anaconda pip

# install PyTorch - https://pytorch.org/get-started/locally/
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt
```

To download datasets: 

* Extract all data to desired path: [Datasets](https://queensuca-my.sharepoint.com/:f:/g/personal/19nyk1_queensu_ca/EuoCEg-JUB9AuqkHWoh8WLQBDWGf3cnJLKzBJGsTkLpqsw?e=inwedL)

Dataset A is a labeled dataset, new dataset is unlabeled. 

## Training

To train the model(s) in the paper, separate tasks can be chosen using flags. Make sure the datasets are extracted in the current directory: `cd path_to_datasets`

### Unsupervised Cell-level Clustering on labeled dataset:
```train
python /path_to_utils/nu_gan.py --task 'cell_representation'
```

* To change the number of clusters:
```train
python /path_to_utils/nu_gan.py --task 'cell_representation' --dis_category 4
```
The default number of clusters (`dis_category`) is 5. 

### Unsupervised Cell-level Clustering on unlabeled dataset:
```train
python /path_to_utils/nu_gan.py --task 'cell_representation_unlabeled'
```

* To change the number of clusters:
```train
python /path_to_utils/nu_gan.py --task 'cell_representation_unlabeled' --dis_category 4
```
The default number of clusters (`dis_category`) is 5. 

### Cell Classification:
```train
python /path_to_utils/nu_gan.py --task 'cell_classification' --experiment_id 123456
```

`experiment_id` identifies the experiment from cell-level clustering to perform cell classification. Can be retrieved from the output folder name from `cell_representation` or `cell_representation_unlabeled`.

Other hyperparameters that don't have specified flags can be changed in `nu_gan.py`.

## Evaluation

To evaluate the model on labeled data:

```eval
python /path_to_utils/eval.py --model_path "path_to_models" --dis_category 5
```

The default number of clusters (`dis_category`) is 5. 

## Pre-trained Models

You can download pretrained models here:

- [Unsupervised Cell-level Clustering on labeled dataset](https://queensuca-my.sharepoint.com/:f:/g/personal/19nyk1_queensu_ca/EqGlvyQmEPFHtupKWOExlMMBy7XFn075GPAoqu9yvqBYaA?e=lTlKL8) trained on the labeled Dataset A. 
- [Unsupervised Cell-level Clustering on unlabeled dataset](https://queensuca-my.sharepoint.com/:f:/g/personal/19nyk1_queensu_ca/EpWxSHIduaNOsPusqVZy9TEB0QCeXwnytO4nRzDy0RSPCg?e=h5hdCl) trained on the unlabeled new dataset. 

## Results

Our model achieves the following performance on :

### Unsupervised Cell-level Clustering on labeled dataset

|          | Purity  | Entropy | F-score |
| ------------------ |---------------- | -------------- | -------------- |
| Reproduction   |     0.803         |      0.914      |      0.810      |

![datasetA_plot](/datasetA_plot.png)

### Unsupervised Cell-level Clustering on unlabeled dataset

|    Clusters      |  V(D, G) | Lq | 
| ------------------ |---------------- | -------------- | 
| 5   |     -6.08         |      0.031      |  

Where V(D, G) is the value function of the discriminator and generator networks and Lq is the loss of the auxiliary network. 

<img src=/new_dataset_plot.png width="580" height="270">

## Contributing

If you'd like to contribute, or have any suggestions, you can contact us at nathalia.yun@gmail.com or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
