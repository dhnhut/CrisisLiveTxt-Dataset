# CrisisLiveTxt-Dataset - A Large-Scale Crisis-related Dataset for Humanitarian Information Detection on Social Media

> We briefly describe our work for a non-technical audience to promote the AI for good purpose of the project.
>
> For expert and scientific readers, we will publish our article after proceeding with the conference venue.

## A. Overview

Detecting and classifying crisis-related event information on social media is crucial for humanitarian operations. Our work is building a dataset that simulates real-world scenarios and creating a pipeline for the task by creating a Large-Scale dataset and a Baseline classifier:

### A.1. The CrisisLiveTxt Dataset

- Gathering and selecting 158,975 records from 14 previous crisis-related annotated datasets.
- Mix with another 11 noise datasets from Twitter, which contain a similar topic to the crisis-related dataset, then the total number of datasets is 3,605,776 records.
- We applied semantic similarity for a high-quality dataset by preventing data leakage.

### A.2. The Cascade Classifier Pipeline

- We apply cascading classification by fine-tuning BERT and Llama models for a multi-stage classifier. Technique for Imbalanced Data and Parameter-Eﬀicient Fine-Tuning
(PEFT) were applied, such as FocalLoss and QLoRA.

> We cannot publish the dataset temporarily because we are in the publication process and subject to the [X Developer Agreement](https://docs.x.com/developer-terms/agreement).



## B. The datasets

Few Distributions of Class in the Dataset are in Figures belows:

### B.1. Informativeness and Noise

![The distribution of crisis-related annotated tweet and noise.](./assets/figures/annotated_noise_distribution.png)

#### Figure 1: The Distribution of Crisis-related annotated tweets and Noise

- Informativeness means whether the tweet contains useful information to a crisis or not and annotated by human.
- Noise means tweet that is not related or close topic (i.e. weather)

### B.2. Humanitarian Informative Category

![The distribution of crisis-related annotated tweet and noise.](./assets/figures/informativeness_distribution_by_humanitarian_category.png)

#### Figure 2: The Classes Distribution of Humanitarian Crisis-related tweets

The challeges of the task is how to preciesly detect the Humanitarian Information from a mass amount of Noise and Related but Non-Informative and How to accurate classify to the correct category.

To demonstrate the issue, let's consider our dataset. Fore every informative tweet has another 40 non-informative. If the model simply classify all tweets is non-informative, the model correcly classify 40 non-negative and wrong at only one record, then the accuracy is $40/41 = 97.56\%$. Result is really high but totally not usefull for the task.

### B.3. Cleaning steps

The crucital step of any AI model is data processing. Beside basic cleaning step such as de-duplication, we also apply semantic filtering to remove similar tweet from training and evaluating process to prevent [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)). Think about a student who practised a question which will appeared again in the exam, he can easily solve it but we can not say he know how to solve or just remember the  answer.

And we not simple filter by `Lexical` (word comparision) but the `Semantic` meaning. For example:

#### Table 1. The Classes Distribution of Humanitarian Crisis-related tweets

| No. | Sentence | Lexical | Semantic |
| -: | :- | :- | -: |
| 1 | Help! Trapped on the second floor! #flood | - | - |
| 2 | The bird is trapped on the second floor. | 0.754 × | 0.444 |
| 3 | I am stuck upstairs. Need rescue from the flood! | 0.128 | 0.729 ✓ |

The higher number, is higher similarity. If we compare the sentence `1` with sentence `2` and `3`. If we only filter by compare words exactly (lexical) the output is not deserved.

To do that in the large scaled data and leverage GPU optimization, we apply [FAISS](https://github.com/facebookresearch/faiss), a library from META for efficient similarity search and clustering of dense vectors.

## C. The pipleline

![Testing pipeline](./assets/figures/testing_pipeline.png)

#### Figure 3: The Classes Distribution of Humanitarian Crisis-related tweets
