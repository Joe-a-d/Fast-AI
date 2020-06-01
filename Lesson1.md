---
tags: [fast.ai, ML, mooc, Notebooks/ML]
title: Lesson1
created: '2020-05-30T18:47:48.108Z'
modified: '2020-06-01T03:54:26.718Z'
---

# Intro


## General Recommendations for success

- Do not try to understand everything during the lecture, make sure to take note of unclear points for clarification during review

- 70/80h of work expected: 8 self + 1.5 lecture
-- expected deadline: 4x/w :: Mid-august ; 3x/w :: Mid-Sep

- the course will be project-drive, first we code and then we delve into theory

## Projects

- simple photos (pets) classification
- text corpus analysis (movie review sentiment)
- supermarket sales predictor
- recommender system (movies)


# Lecture

We are going to be using both [kaggle](https://kagle.com) datasets and academic datasets

[fastai](https://docs.fast.ai/)

`fastai` is a python library which sits on top of pyTorch

`fastai` documentation provides types for arguments, which lets you know what a method expects as argumets

```
untar_data(url: str, fname: Union[pathlib.Path, str] = None, dest: Union[pathlib.Path, str] = None, data=True, force_download=False, verbose=False) -> pathlib.Path
```

We read `Union` in the context of sets , i.e as `"or"`. Hence, we can see that the function takes *3 arguments* an `url` of type string, `fname` and `dest` which can be either a *path* or a *string*  and which are optional as indicated by `= None`


## Inspecting the data

- In data science the first step is always to have a look at the data, i.e understanding how the data directories are structured, what the labels are and what some sample images look like

- Labels [^label] 
[^label]: In ML a label is whatever it is that we're trying to predict

- A shortcoming of GPU technology is that in order for it to be fast it needs to perform the same instruction to similar objects. In this case, having different sized images will prevent it from performing appropriately, hence we specify a uniform size when processing the images

  - industry standards : 224x224

- Recall also from *Yang* that for it is important that we *normalize* [^normal] the data for optimal performance

[^normal]: *normalization* is used somewhat informally in statistics, and so can take many meanings. In more formal settings it tends to involve transforming data using a *`z/t-score`. Often in ML, it is equivalent to *feature scalling*, i.e getting rid of units to better compare data from different places. This is usually done by converting the data's range to fall between [0,1]. [more](https://www.statisticshowto.com/normalized/)

- After cleaning up your data, perform some sanity checks by inspecting it further. Does the total amount of labels match your array? Do the photos match their labels?

## Training

- The `learner`[^learner] is a general type of object for things that can learn, similar to like a `Databunch` is for data. It knows about 2 things, what's your data and  what's your model.

[^learner]: A *learner* or *ML Algorithm* is the actual program whose main goal is to generalize from experience, i.e. from data fed into it.

- In this lesson we'll train our model using a `Convolutional Neural Network`[^CNN] .

[^CNN]: A *ConvNev*  is specific type of artificial neural network that uses perceptrons, a machine learning unit algorithm, for supervised learning, to analyze data. It is most commonly used for image and NLP.

`learn = cnn_learner(data, models.resnet34, metrics=error_rate)`

Note that the `resnet34` provides us with `pre-trained weights` [^pre_weights] , this increases the efficiency of our learner. In this particularly case, the model has been trained by looking at thousands of images with 1000 different labels, so our model knows that there could be a thousand categories of things in an image. This notion of *"seeding"* a model is an integral part of modern ML, and is refered to as `transfer-learning` [^transfer_learning]. 

This practice can increase the efficiency 100-fold. Though when reducing the amount of data fed into the model, it is important to consider `overfitting` [^overfitting] and use `validation sets` [^validation_set] , as we'll see in future lessons


[^transfer_learning]: *TL* focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.  

[^pre_weights]: A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task. The *pretrained weights* are just the product of the trained model, which determine how strong each neural connection should be.

[^overfitting]: *Overfitting* is a modeling error that occurs when a function is too closely fit to a limited set of data points. In reality, the data often studied has some degree of error or random noise within it. Thus, attempting to make the model conform too closely to slightly inaccurate data can infect the model with substantial errors and reduce its predictive power. [more](https://www.investopedia.com/terms/o/overfitting.asp)

[^validation_set]: A *validation set* or *dev set* is a set of data used during training with the goal of finding and optimizing the best model to solve a given problem. It usually makes up about 20% of the total data, in the mid-stage of training, and it is used for parameter tunnig and overfitting checks. [more](https://whatis.techtarget.com/definition/validation-set)


 ## Results

`ClassificationInterpretation` object

After training our model the next stage is the dissection of the results. Use the fastai tools to interpret any unexpected results. See if the errors make sense. Does any particular label standout? Ask questions and investigate

A *loss function* essentially tells you how good your prediction is. `top_losses` a fastai method, helps us dissect the results by outputting a tupple consisting of `(prediction, actual, loss, probability of actual)`

A *confusion matrix* for $n$ labels is a $n \times n$ matrix composed of all the labels with the expected value on one axys and the actual on the other and its entries are the output of the model for each iteration. `most_confused` performs a similar analysis, but it prints a list of tupples of the labels with the greates number of errors `(expected, actual, #errors)` .

## Tuning

Running a *fit* learner on a pre-trained model will just add a few layers to the end of that model and train those. We can run unfreeze so that it runs on the whole model. 

Each layer's patterns increase in (semantic) complexity. When fine-tunning we really don't want to change the upper layer, since they are the basic building blocks (lines, circles, etc.). 

If we just run another *fit* cycle, the error rate will increase, since it will train all the layers at the same speed, so it deems the more complex and "*well-defined"* patterns as important as the basic building blocks.

run find and inspect the results. this allows you to see the range of layers which performed the worst, and adjust the learning rate for each accordingly. @1:23 (more on this next week)


# To-Do

- build your own image dataset
- go over lesson1.ipynb
- anki


## Jupyter/Python/fastai

- `help()` shows the doc for a function, alternative you can use tab autocompletion and `shift-Tab` for full docs, or `doc()` for an hyperlink to the official docs.
- [py3 Path objects](https://docs.python.org/3/library/pathlib.html)
- [`DataBunch`](https://docs.fast.ai/basic_data.html#DataBunch)
- [`Learner`](https://docs.fast.ai/basic_train.html#Learner)

## Practical Tips

- Finding labels from datasets - Conventions:
  1. Labeled folders / Imagenet style : use `from_folder`
  2. csv file with labels : use `from_csv`
  3. label in filename : use regex `from_name_re` or a function `from_name_func`
  
  For more see [docs](https://docs.fast.ai/vision.data#ImageDataBunch) for `ImageBunch` methods

<p style="font-size:2em"> Glossary </p>


