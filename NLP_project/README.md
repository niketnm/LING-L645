# NLP debias word embedding project

This project aims to replicate and build upon the findings of the paper "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings." This paper discusses how biases can be present in the data used to train machine learning models and how these models can amplify these biases. The project specifically focuses on word embedding, a common method of representing text data as vectors, and how this can contribute to bias in machine learning models. We aim to address this issue by finding ways to reduce or eliminate these biases in the word embedding process.



# Dataset
**w2vNEWS (Word2Vec embedding trained on a corpus of Google news texts) - w2v_gnews_small.txt**
1. This is a word embeddings trained on Google News articles which exhibit female/male gender stereotypes to a disturbing extent. This raises concerns because their widespread use, as will be described, often tends to amplify these biases. 
2. 300-dimensional word2vec embedding, which has proven to be immensely useful since it is high quality, publicly available, and easy to incorporate into any application. In particular, we downloaded the pre-trained embedding on the Google News corpus,4 and normalized each word to unit length as is common. 
3. Starting with the 50,000 most frequent words, we selected only lower-case words and phrases consisting of fewer than 20 lower-case characters (words with upper-case letters, digits, or punctuation were discarded). 
4. After this filtering, 26,377 words remained. While the focus is on w2vNEWS, we show later that gender stereotypes are also present in other embedding data-sets.

# Debias Algorithm
The paper proposes methods to reduce the impact of bias in word embeddings while preserving their useful properties, such as the ability to cluster related concepts and solve analogy tasks. The first method involves identifying the gender bias subspace using Principal Component Analysis on gender pair difference vectors. The second method, called "neutralize and equalize," removes gender-neutral words from the gender subspace and makes them equidistant outside of the subspace. The third method, called "soft bias correction," reduces differences between certain sets of words while maintaining as much similarity to the original embedding as possible. This method allows for control over the trade-off between bias reduction and preservation of useful properties.

# Results
**Figure 1. Gender Bias - Word Embedding Space**
![alt text](https://github.com/niketnm/LING-L645/blob/main/NLP_project/results/english_gender_debiased/debiasFull.png)
**Figure 2. Racial Bias - Word Embedding Space** 
![alt text](https://github.com/niketnm/LING-L645/blob/main/NLP_project/results/english_racial_bias/racialBias.png)

**Figure 3. Debiased Word Embedding Space after algorithm** 
![alt text](https://github.com/niketnm/LING-L645/blob/main/NLP_project/results/english_gender_debiased/Debiased.png)
