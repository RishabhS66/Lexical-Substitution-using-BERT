#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow
from tensorflow import keras
import string

import gensim
import transformers 

from typing import List

def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    stop_words = set(stopwords.words('english'))
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    tokens = s.split()
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    syn_sets = wn.synsets(lemma, pos=pos)

    candidates = set()
    for syn_set in syn_sets:
        for syn_lemma in syn_set.lemmas():
            candidate = syn_lemma.name().replace('_', ' ')
            # Check candidate is not the input lemma
            if candidate != lemma:
                candidates.add(candidate)

    candidates_list = list(candidates)
    return candidates_list

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    candidates = get_candidates(context.lemma, context.pos)

    frequency_dict = {}
    for candidate in candidates:
        total_frequency = 0
        for synset in wn.synsets(context.lemma, context.pos):
            for syn_lemma in synset.lemmas():
                if candidate == syn_lemma.name().replace('_', ' '):
                    total_frequency += syn_lemma.count()
        frequency_dict[candidate] = total_frequency

    predicted_synonym = max(frequency_dict, key=frequency_dict.get)

    return predicted_synonym

def wn_simple_lesk_predictor(context : Context) -> str:
    synsets = wn.synsets(context.lemma, context.pos)
    target_context = set(context.left_context + [context.lemma] + context.right_context)

    max_score = 0
    best_replacement = None

    for synset in synsets:
        synset_content = set()

        synset_content.update(tokenize(synset.definition()))
        for example in synset.examples():
            synset_content.update(tokenize(example))

        for hypernym in synset.hypernyms():
            synset_content.update(tokenize(hypernym.definition()))
            for example in hypernym.examples():
                synset_content.update(tokenize(example))

        overlap_score = len(synset_content & target_context)
        synset_target_score = sum([target.count() for target in synset.lemmas()
                                   if target.name().replace('_', ' ') == context.lemma])
        net_score_now = 1000*overlap_score + 100*synset_target_score
        for candidate in synset.lemmas():
            if candidate.name().replace('_', ' ') != context.lemma:
                score = net_score_now + candidate.count()
                if score > max_score:
                    max_score = score
                    best_replacement = candidate.name().replace('_', ' ')

    return best_replacement
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        target_word = context.lemma
        pos = context.pos

        synonyms = get_candidates(target_word, pos)
        max_sim_sc = -1
        best_synonym = None
        for synonym in synonyms:
            if synonym in self.model:
                score = self.model.similarity(target_word, synonym)
                if score > max_sim_sc:
                    max_sim_sc = score
                    best_synonym = synonym

        return best_synonym


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)

        input_text = context.left_context + ['[MASK]'] + context.right_context
        input_toks = self.tokenizer.encode(input_text)
        input_mat = np.array(input_toks).reshape((1, -1))

        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]

        masked_position = input_toks.index(self.tokenizer.mask_token_id)

        best_candidate_bert = np.argsort(predictions[0][masked_position])[::-1]
        best_candidate_bert = self.tokenizer.convert_ids_to_tokens(best_candidate_bert)
        best_synonym = ""
        for synonym in best_candidate_bert:
            if synonym in candidates:
                best_synonym = synonym
                break
        return best_synonym

class NewPredictor(object):

    def __init__(self, filename):
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def bert_score(self, context):
        input_text = context.left_context + ['[MASK]'] + context.right_context
        input_toks = self.tokenizer.encode(input_text)
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.bert_model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        masked_position = input_toks.index(self.tokenizer.mask_token_id)
        return predictions[0][masked_position]

    def word2vec_similarity(self, target_word, candidate):
        try:
            return self.word2vec_model.similarity(target_word, candidate)
        except KeyError:
            return -1.0  # Handle the case where the word is not in the Word2Vec model

    def prediction(self, context: Context) -> str:

        """
        MY APPROACH:
        We get some top choices of words from BERT which could possibly replace the target word, keeping the context
        under consideration.  Now, we also add the candidate words from WordNet. From the list of words that we have
        till now, we choose the word which is the most similar to the target word (word2Vec similarity).
        """

        bert_scores = self.bert_score(context)
        bert_scores_idx = np.argsort(bert_scores)[::-1]
        bert_candidates = (self.tokenizer.convert_ids_to_tokens(bert_scores_idx)[:min(45, len(bert_scores_idx))]
                           + get_candidates(context.lemma, context.pos))
        candidates = [x for x in bert_candidates if x != context.lemma and x.lower() != context.word_form.lower()]
        candidates = list(set(candidates))

        ranked_candidates = sorted(candidates,
                                   key=lambda x: self.word2vec_similarity(context.lemma, x), reverse=True)
        return ranked_candidates[0]

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # predictor = BertPredictor()

    myPredictor = NewPredictor(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging

        # prediction = wn_frequency_predictor(context)

        # prediction = wn_simple_lesk_predictor(context)

        # prediction = predictor.predict_nearest(context)

        # prediction = predictor.predict(context)

        prediction = myPredictor.prediction(context)

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
