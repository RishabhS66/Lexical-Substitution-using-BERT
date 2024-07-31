# Lexical Substitution Using BERT

```
Author: Rishabh Srivastava
```

## Overview

This project focuses on enhancing lexical substitution by combining the strengths of WordNet, Word2Vec embeddings, and BERT. 

Lexical substitution aims to find the most appropriate replacement for a target word in a given context, preserving the original meaning of the sentence. The integration of these three resources ensures better contextual understanding and semantic similarity, resulting in more accurate substitutions.

## Project Structure
The main script for this project is `lexsub_main.py`, which implements various predictors for the lexical substitution task. The predictors use different strategies ranging from simple frequency-based methods to advanced neural network models.

### Files
- `lexsub_main.py`: Main script that executes the lexical substitution task using different predictors.
- `lexsub_xml.py`: XML parser to read input data and convert it into Python objects.
- `lexsub_trial.xml`: Sample input data containing sentences with target words for substitution.
- `gold.trial`: Gold standard annotations for the trial data, used for evaluation.
- `score.pl`: Scoring script to evaluate the performance of the predictors.

## Dependencies
The following Python packages are required:

- `nltk`
- `tensorflow`
- `gensim`
- `transformers`
- `numpy`

Ensure you have downloaded the necessary NLTK data:
```
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```

## Predictors

The following prediction strategies were implemented for analysis:

1. **WordNet Frequency Predictor**: Predicts the synonym with the highest occurrence frequency from WordNet.
2. **WordNet Simple Lesk Predictor**: Uses the Lesk algorithm for Word Sense Disambiguation to find the most appropriate synonym.
3. **Word2Vec Substitution**: Utilizes pre-trained Word2Vec embeddings to find the most semantically similar word.
4. **BERT Predictor**: Uses a pre-trained BERT model to predict the best substitute based on the context.
5. **New Predictor Proposed**: A novel fusion strategy combining BERT's contextual predictions with Word2Vec's semantic similarity and WordNet's lexical relations. We start by obtaining some top choices of words from BERT that could potentially replace the target word while preserving the context. Next, we enhance this list by adding candidate words from WordNet. From this expanded list, we then select the word that is most similar to the target word based on Word2Vec similarity.

## Running the Script
To run the script and get predictions for the trial data, use the following command:
```
python lexsub_main.py lexsub_trial.xml  > {FILENAME}.predict 
```
This will output the predicted substitutions for each target word in the context.

Then, run the scoring script (written in perl) on the `.predict` file:
```
perl score.pl {FILENAME}.predict gold.trial
```

## Results for the New Predictor
The project achieved precision and recall of **0.189** on 206 attempted instances with mode-specific scoring. The integration of BERT, Word2Vec, and WordNet demonstrated a significant improvement in predicting contextually appropriate substitutes.
