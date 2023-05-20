# Using Automatic Text Reduction Techniques for Annotating News Portal Articles.

## Abstract
Automatic text reduction, a.k.a. text summarization, is one of the most common sequence-to-sequence tasks in the field of Natural Language Processing (NLP). It is a challenging problem that involves reducing the length of a given text while retaining its most important information. This work reviews various methods for text summarization, including extractive and abstractive approaches, and provides an overview of different deep learning architectures that have been applied to this task, as well as user interface implementation tools and backend technologies. The work then focuses on the training of several neural network models for Russian texts, using a dataset of news articles from the High School of Economics news platform, and analyzing their results with different text quality metrics. The final stage of the work involves deploying the resulting text summarization system to the university’s website.

Full dimploma paper: [Here (in Russian)](https://github.com/paulkulpin/textsum_HSE_news/blob/main/TextSum_HSEnews_diploma.pdf)

#To run the backend file:

```bash
sudo docker build -t "sum_text_hse_ml" -f dockerfile . 
```
