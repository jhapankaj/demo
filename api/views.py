
# from nltk.corpus import stopwords
# from stop_words import get_stop_words
# # from fuzzywuzzy import fuzz
# # from tensorboard import summary
# # from tensorboard import summary
# # import yake
# # import pytextrank
# # import spacy
# # from google.cloud import speech
# import math
# import os
# from email.mime import audio
# from pydub.utils import make_chunks
# from pydub import AudioSegment
import json
from skimage import color
from scipy.interpolate import interp1d
from scipy import interpolate
from pylab import *
import random
import matplotlib.pyplot as plt
import sys
import dlib
from rest_framework.parsers import MultiPartParser, FormParser
from api.custom_renderer import PNGRenderer, JPEGRenderer
# import pprint
# from elasticsearch_dsl import Search
# from elasticsearch_dsl import Q as esQ
# from elasticsearch import Elasticsearch
from api.serializers import (
    recievemessageSerializer, recievereplySerializer, QuizQuestionSerializer,
    AnswerOptionSerializer, chatRecommendationSerializer, customerResponseQuizSerializers,
    ingredientSerializer, IngredientRecommendationSerializer, allProductSerializer, FoundationImageSerializer,
    ExtractSkinSerializer, TryOnsSerializers, TryonsImageSerializer)
from api.models import (
    customer_chat, SpreeQuestions, SpreeAnswerOptions, ApiProducts, ApiAllProductUniq,
    customerquizresponse, ApiQuizquestionsNew, ApiAllProductUniq, SpreeTaxons, ApiProductvectorNew,
    SpreeProducts, foundation_image, foundation_product_options, tryons_image
)
from django.shortcuts import render
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404
from django.conf import settings
from django.db.models import Q
from rest_framework import status
from rest_framework.parsers import FileUploadParser
import re
# from nltk.stem import WordNetLemmatizer
# from nltk.stem.lancaster import LancasterStemmer
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.tag.stanford import StanfordNERTagger
# from nltk.corpus import words as dict_words
# import numpy as np
import pandas as pd
# import pickle
# from fuzzywuzzy import fuzz
# import nltk
from functools import reduce
import operator
import itertools
# import mediapipe as mp
# from numba import jit
# import cv2
# from mtcnn.mtcnn import MTCNN
from sklearn.cluster import KMeans
from collections import Counter
# from keras.models import load_model

# from sqlalchemy import create_engine
# import datetime

import cv2
import mediapipe as mp
import numpy as np
# # import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# engine = create_engine(
#     "mysql+pymysql://root:root@localhost/demo")
# cur = engine.connect()

# stop_words = set(stopwords.words('english'))

# taxon_all = []
# taxon_q = cur.execute("select name from spree_taxons").fetchall()

# st = StanfordNERTagger(settings.MEDIA_ROOT+'/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
#                        settings.MEDIA_ROOT+'/stanford-ner-2018-10-16/stanford-ner.jar', encoding='utf-8')


# Create your views here.


# Post messagfee to the data base : post api

class recievemessageView(APIView):
    # serializer_class = recievemessageSerializer
    # permission_classes = (IsAuthenticated,)

    def get(self, request, format=None):
        chat = customer_chat.objects.all()
        serializer = recievemessageSerializer(chat, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        print(request.data, "dhdhdh")
        serializer = recievemessageSerializer(data=request.data)
        print(serializer)
        if serializer.is_valid():
            print(request.data, "kfkfkfkkf")
            if request.data['question'] == '' and request.data['sessid'] == '':
                print(1)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                print(2)
                serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#     # For getting response from the bot
# lemmatizer = WordNetLemmatizer()
# stemmer = LancasterStemmer()
# #  Model to predict it the message is spam / profanity
# # words = pickle.load(open(settings.MEDIA_ROOT+'/r_words.pkl', 'rb'))
# # classes = pickle.load(open(settings.MEDIA_ROOT+'/r_classes.pkl', 'rb'))
# # rmodel = load_model(settings.MEDIA_ROOT+'/r_model.h5')

# # #  Models to predict customer Service
# # words_cs = pickle.load(open(settings.MEDIA_ROOT+'/words_cs.pkl', 'rb'))
# # labels_cs = pickle.load(open(settings.MEDIA_ROOT+'/classes_cs.pkl', 'rb'))
# # csmodel = load_model(settings.MEDIA_ROOT+'/cs_model1.h5')


# # words_care = pickle.load(open(settings.MEDIA_ROOT+'/words_care.pkl', 'rb'))
# # classes_care = pickle.load(open(settings.MEDIA_ROOT+'/classes_care.pkl', 'rb'))
# # model = load_model(settings.MEDIA_ROOT+'/skincare_model.h5')


# # words_t = pickle.load(open(settings.MEDIA_ROOT+'/words_t.pkl', 'rb'))
# # classes_t = pickle.load(open(settings.MEDIA_ROOT+'/classes_t.pkl', 'rb'))
# # skintypemodel = load_model(settings.MEDIA_ROOT+'/skintype_model.h5')


# # words_concern = pickle.load(
# #     open(settings.MEDIA_ROOT+'/words_concern.pkl', 'rb'))
# # classes_concern = pickle.load(
# #     open(settings.MEDIA_ROOT+'/classes_concern.pkl', 'rb'))
# # skinconcernmodel = load_model(settings.MEDIA_ROOT+'/skinconcern_model.h5')


# def brand_search(message):
#     brand_search_q = cur.execute(
#         "select title , slug , id  from spree_brands order by title").fetchall()
#     # all_brands = []
#     brand_found = None
#     res = None
#     brands_given_to_customer = []
#     for brand in brand_search_q:
#         if res is None:
#             if brand[0].lower() in message.lower():
#                 brand_found = brand[0]
#                 res = "You can find all products from " + brand_found + \
#                     " here:<a href='https://www.yuty.me/c/all?brands=" + \
#                     str(brand[2])+"'>" + brand_found + \
#                     "</a> <p> Would you like to continue, Yes or No ?"
#                 break
#             elif brand[0].lower() not in message:
#                 # customer text matching
#                 tokens = word_tokenize(message.lower())
#                 customer_text_list = [
#                     i for i in tokens if not i in stop_words and not i == "+" and not i == "beauty" and not i == "london"]
#                 for word in customer_text_list:
#                     if word.lower() not in taxon_all:
#                         if len(word) > 2:
#                             partial_ratio = fuzz.partial_ratio(
#                                 word, brand[0].lower())
#                             if partial_ratio >= 90 and (brand[0].lower() not in brands_given_to_customer):
#                                 brands_given_to_customer.append(brand[0])
#                 if len(brands_given_to_customer) > 0:
#                     brand_found = brands_given_to_customer[0]
#                     res = "You can find all products from " + brand_found + \
#                         " here:<a href='https://www.yuty.me/c/all?brands=" + \
#                         str(brand[2])+"'>" + brand_found + \
#                         "</a > <p > Would you like to continue, Yes or No ?"
#     return res


# def ingredient_search(message):
#     glossary_res = None
#     print("starting ingred search")
#     # TESTING ---------------------------------------
#     tokens = word_tokenize(message.lower())
#     customer_text_list = [
#         i for i in tokens if not i in stop_words and not i == "+" and not i == "beauty"]
#     # END OF TESTING ---------------------------------------

#     for i in range(len(chatbot_voc)):
#         if str(chatbot_voc.iloc[i][0]).lower() in message.lower():
#             if str(chatbot_voc.iloc[i][1]) != 'empty':
#                 glossary_res = str(chatbot_voc.iloc[i][1])
#                 # Add Recommendation of all products containing ingredients
#                 ingred_prods = all_product_uniq.objects.all().filter(
#                     ingredients__icontains=chatbot_voc.iloc[i][0])
#                 ingredient_list = []
#                 if len(ingred_prods) > 0:
#                     for product in ingred_prods:
#                         # print(product)
#                         ingredient_list.append(
#                             '<p> <a href="https://www.yuty.me/p/'+str(product.product_id)+'/'+product.slug+'">' + product.product_name + '</a></p>')
#                 # tmp = glossary_res + ' Please find relevant products below: ' + \
#                 #     chatbot_voc.iloc[i][0] + '' + \
#                 #     ' '.join(ingredient_list[0:3])
#                 tmp = glossary_res + ' Please find relevant products below: ' + \
#                     ' '.join(ingredient_list[0:3])
#                 glossary_res = tmp + '<p> Would you like to continue, Yes or No ?'
#             elif str(chatbot_voc.iloc[i][1]) == 'empty':
#                 glossary_res = 'The scientists and engineers behind Yuty will be adding this definition soon, thank you 🚀'
#         else:
#             # TESTING ---------------------------------------
#             for word in customer_text_list:
#                 if len(word) > 2:
#                     partial_ratio = fuzz.partial_ratio(
#                         str(chatbot_voc.iloc[i][0]).lower(), word)
#                     if partial_ratio > 90:
#                         # print(str(chatbot_voc.iloc[i][0]).lower())
#                         glossary_res = str(chatbot_voc.iloc[i][1])
#                         ingred_prods = all_product_uniq.objects.all().filter(
#                             ingredients__icontains=chatbot_voc.iloc[i][0])
#                         ingredient_list = []
#                         if len(ingred_prods) > 0:
#                             for product in ingred_prods:
#                                 # print(product)
#                                 ingredient_list.append(
#                                     '<p> <a href="https://www.yuty.me/p/'+str(product.product_id)+'/'+product.slug+'">' + product.product_name + '</a></p>')
#                         tmp = glossary_res + ' Please find relevant products below: ' + \
#                             ' '.join(ingredient_list[0:3])
#                         glossary_res = tmp + '<p> Would you like to continue, Yes or No ?'

#             # END OF TESTING ---------------------------------------
#     return glossary_res


# def product_category_search(customer_text):
#     result = None
#     pred_tag = None
#     if ('product' in customer_text or 'looking' in customer_text) and 'skin' in customer_text and 'hair' in customer_text:
#         result = "You can find all products listed on Yuty.me here: <a href='https://www.yuty.me/c/all'>List of Products</a> <p> Would you like to continue, Yes or No ?"
#         pred_tag = 'Product'
#     elif ('product' in customer_text or 'looking' in customer_text) and 'skin' in customer_text:
#         result = "You can find all skincare products here: <a href='https://www.yuty.me/c/skincare'>List of Skincare Products</a> <p> Would you like to continue, Yes or No ?"
#         pred_tag = 'Product'
#     elif ('product' in customer_text or 'looking' in customer_text) and 'hair' in customer_text:
#         result = "You can find all haircare products here: <a href='https://www.yuty.me/c/haircare'>List of Haircare Products</a> <p> Would you like to continue, Yes or No ?"
#         pred_tag = 'Product'
#     elif ('product' in customer_text or 'looking' in customer_text) and ('body' in customer_text or 'bath' in customer_text):
#         result = "You can find all body & bath products here: <a href='https://www.yuty.me/c/bath-and-body'>List of Bath & Body Products</a> <p> Would you like to continue, Yes or No ?"
#         pred_tag = 'Product'
#     elif ('product' in customer_text or 'looking' in customer_text) and ('makeup' in customer_text or 'make up' in customer_text):
#         result = "You can find all makeup products here: <a href='https://www.yuty.me/c/makeup'>List of Makeup Products</a> <p> Would you like to continue, Yes or No ?"
#         pred_tag = 'Product'
#     elif ('product' in customer_text or 'looking' in customer_text) and 'fragrance' in customer_text:
#         result = "You can find all fragrance here: <a href='https://www.yuty.me/c/fragrance'>List of Fragrance</a> <p> Would you like to continue, Yes or No ?"
#         pred_tag = 'Product'
#     elif ('product' in customer_text or 'looking' in customer_text) and ('gift' in customer_text):
#         result = "You can find all gift ideas here: <a href='https://www.yuty.me/c/gifts'>List of Gift Ideas</a> <p> Would you like to continue, Yes or No ?"
#         pred_tag = 'Product'

#     if result is None:
#         if 'exfoliator' in customer_text or 'exfoliant' in customer_text or 'scrub' in customer_text:
#             result = "Yuty is happy to provide a list of all Exfoliator products here: <a href='https://www.yuty.me/c/skincare?categories=76,86&price=0,500&page=1'>Exfoliator</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#             pred_tag = 'Product Category'
#         elif 'serum' in customer_text or 'elixir' in customer_text or 'potion' in customer_text or 'treatment' in customer_text:
#             result = "Yuty is happy to provide a list of all Serum products here: <a href='https://www.yuty.me/c/skincare?categories=76,77&price=0,500&page=1'>Serum</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#             pred_tag = 'Product Category'
#         elif 'moisturiser' in customer_text or 'moisturizer' in customer_text or 'lotion' in customer_text:
#             result = "Yuty is happy to provide a list of all Moisturiser products here: <a href='https://www.yuty.me/c/skincare?categories=76,87,131,822,122,123,89,821,823,77,824,78&price=0,500&page=1'>Moisturiser</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#             pred_tag = 'Product Category'
#         elif 'mask' in customer_text or 'clay' in customer_text or 'face peel' in customer_text or 'masque' in customer_text or 'peel' in customer_text:
#             result = "Yuty is happy to provide a list of all Mask products here: <a href='https://www.yuty.me/c/skincare?categories=76,343,91,432,79,83,825,827,86,85,262&price=0,500&page=1'>Mask</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#             pred_tag = 'Product Category'
#         elif 'eye cream' in customer_text or 'eye' in customer_text or 'anti wrinkle' in customer_text or 'dark circles' in customer_text or ('bags' in customer_text and 'eye' in customer_text):
#             result = "Yuty is happy to provide a list of all Eye products here: <a href='https://www.yuty.me/c/skincare?categories=76,132,396,124,443,80,831,832,829,830,828&price=0,500&page=1'>Eyes</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#             pred_tag = 'Product Category'
#         elif 'antiaging' in customer_text or 'anti-aging' in customer_text or 'anti aging' in customer_text:
#             result = "Yuty is happy to provide a list of all Anti-aging products here: <a href='https://www.yuty.me/c/skincare?categories=76,83&price=0,500&page=1'>Anti-aging</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#             pred_tag = 'Product Category'
#         elif 'cleanser' in customer_text or 'clarifier' in customer_text or 'disinfectant' in customer_text or 'refiner' in customer_text or 'face wash' in customer_text or 'antibacterial' in customer_text or 'makeup remove' in customer_text or 'clean gel' in customer_text or 'clean lotion' in customer_text or 'clean oil' in customer_text:
#             result = "Yuty is happy to provide a list of all Cleanser products here: <a href='https://www.yuty.me/c/skincare?categories=76,819,383,820,93,818,440,84&price=0,500&page=1'>Cleanser</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#             pred_tag = 'Product Category'
#         elif 'blemishes' in customer_text or 'blemish' in customer_text or 'anti-blemish' in customer_text or 'antiblemish' in customer_text or 'anti blemish' in customer_text:
#             result = "Yuty is happy to provide a list of all Anti-blemish products here: <a href='https://www.yuty.me/c/skincare?categories=76,85&price=0,500&page=1'>Anti-blemish</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#             pred_tag = 'Product Category'
#         elif 'face oil' in customer_text or 'face serum' in customer_text or 'cleansing oil' in customer_text or 'facial serum' in customer_text:
#             pred_tag = 'Product Category'
#             result = "Yuty is happy to provide a list of all Face Oil products here: <a href='https://www.yuty.me/c/skincare?categories=76,87&price=0,500&page=1'>Face Oil</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#         elif ('lip' in customer_text and 'care' in customer_text) or 'lip balm' in customer_text:
#             pred_tag = 'Product Category'
#             result = "Yuty is happy to provide a list of all Lipcare products here: <a href='https://www.yuty.me/c/skincare?categories=76,826,902,429,88&price=0,500&page=1'>Lipcare</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#         elif 'spf' in customer_text or 'sunscreen' in customer_text or 'sun lotion' in customer_text or 'skin protection' in customer_text or ('uv' in customer_text and 'damage' in customer_text):
#             pred_tag = 'Product Category'
#             result = "Yuty is happy to provide a list of all SPF products here: <a href='https://www.yuty.me/c/skincare?categories=76,92,835,834,837,833&price=0,500&page=1'>SPF</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#         elif 'toner' in customer_text or 'astringent' in customer_text or 'mattifier' in customer_text or 'tonic' in customer_text:
#             pred_tag = 'Product Category'
#             result = "Yuty is happy to provide a list of all Toner products here: <a href='https://www.yuty.me/c/skincare?categories=76,93&price=0,500&page=1'>Toner</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#         elif 'mist' in customer_text or 'face spray' in customer_text or 'moist' in customer_text:
#             pred_tag = 'Product Category'
#             result = "Yuty is happy to provide a list of all Facial Mist products here: <a href='https://www.yuty.me/c/skincare?categories=76,131&price=0,500&page=1'>Facial Mist</a> though I must add, unlike other bots I love to make personalised recommendations. You can search for products by brand, concerns, preferences and price <a href='https://www.yuty.me/c/all'>here</a>.<br/>Are you happy to continue chatting with Yuty to receive speedy skincare recommendations?"
#     return {'result': result, 'pred_tag': pred_tag}


# #  Model for Skin care requirement


# def clean_up_sentence(sentence):
#     # tokenize the pattern - split words into array
#     sentence_words = nltk.word_tokenize(sentence)
#     # stem each word - create short form for word
#     sentence_words = [lemmatizer.lemmatize(
#         word.lower()) for word in sentence_words]
#     return sentence_words


# def bow(sentence, words, show_details=True):
#     # tokenize the pattern
#     sentence_words = clean_up_sentence(sentence)
#     # bag of words - matrix of N words, vocabulary matrix
#     bag = [0]*len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 # assign 1 if current word is in the vocabulary position
#                 bag[i] = 1
#                 if show_details:
#                     print("found in bag: %s" % w)
#     return (np.array(bag))


# def bag_of_words(s, words):
#     bag = [0 for _ in range(len(words))]

#     s_words = nltk.word_tokenize(s)
#     s_words = [stemmer.stem(word.lower()) for word in s_words]

#     for se in s_words:
#         for i, w in enumerate(words):
#             if w == se:
#                 bag[i] = 1

#     return np.array(bag)


# def predict_class(sentence, model, words, classes):
#     # filter out predictions below a threshold
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": r[1]})
#     return return_list


# #  Regular Expression prediction like Spam , Thanks , Sorry , Good Bye etc .

# def regularExpression_detection(message):
#     #  use spam detection model as of now its detect spam , thanks etc. Seperate the spam model
#     ints = predict_class(message, rmodel, words, classes)
#     print(ints, "spam")
#     pred_tag = None
#     if ints[0]['probability'] > 0.9999:
#         pred_tag = ints[0]['intent']

#     if pred_tag is None:
#         spam_detected = True
#         msg = message.split(' ')
#         # print(message, message in dict_words.words())
#         for m in msg:
#             if m in dict_words.words():
#                 spam_detected = False
#         if spam_detected == True:
#             pred_tag = 'Spam'
#     return pred_tag


# #  Predict if the intention of Customer Service

# def CustomerService_detection(message):
#     pred_tag = None
#     csresults = csmodel.predict(
#         np.array([bag_of_words(message.lower(), words_cs)]))
#     csresults_index = np.argmax(csresults)
#     csresults = list(csresults[0])
#     cstag = labels_cs[csresults_index]
#     if 'feedback' in message.lower():
#         pred_tag = 'FEEDBACK'
#     elif message.lower() in ('what', 'sorry', 'hmm'):
#         pass
#     elif csresults[csresults_index] > 0.95 and cstag == 'FEEDBACK':
#         pred_tag = 'FEEDBACK'
#     elif csresults[csresults_index] > 0.99999:
#         pred_tag = 'Customer Service - '+labels_cs[csresults_index]
#         print(csresults[csresults_index], pred_tag, 'cs score')

#     return pred_tag

# # Cleaning customer's text for skin regime and skin concern


# def customer_says(customer_text):
#     # customer text cleaning and preprocessing
#     stop_words = set(stopwords.words('english'))
#     customer_text = re.sub(r'\d+', '', customer_text.lower())
#     tokens = word_tokenize(customer_text)
#     customer_text_list = [
#         i for i in tokens if not i in stop_words and not i == "+" and not i == "beauty"]

#     # customer text matching
#     products = ['cleanser', 'clarifier', 'cleaning', 'agent', 'disinfectant', 'refiner', 'refinery', 'soap', 'face', 'wash', 'cleaner', 'clean', 'antibacterial', 'skin', 'makeup', 'remove', 'gel', 'milk', 'lotion', 'oil', 'toner', 'mist', 'astringent', 'mattifier', 'clarifying', 'tonic', 'exfoliant', 'scrub', 'peel', 'serum', 'booster', 'moisturiser', 'complex', 'solution', 'elixir', 'potion', 'treatment', 'day', 'dry', 'cream', 'creme', 'night', 'sunscreen', 'spf', 'sun', 'protection', 'factor', 'protectant', 'sunblock',  'beach',
#                 'eye', 'anti', 'wrinkle', 'dark', 'circles', 'bags', 'repair', 'treatment', 'mask', 'clay', 'sheet', 'masque', 'overnight', 'mud', 'pack', 'oily', 'normal', 'sensitive', 'combination', 'dehydrated', 'irritation', 'acne', 'breakouts', 'redhead', 'blemishes', 'pimples', 'damage', 'puffiness', 'redness', 'scarring', 'pigmentation', 'elasticity', 'dull', 'uneven', 'skintone', 'clogged', 'pore', 'heavy', 'smells', 'fine', 'lines', 'wrinkles', 'blackhead', 'whitehead', 'excess', 'oiliness', 'razor', 'bumps', 'ingrown', 'hairs']

#     # similar words detection
#     similar_words = []
#     similar_products = []
#     partial_ratios_list = []

#     for product in products:
#         for word in customer_text_list:
#             partial_ratio = fuzz.partial_ratio(word, product)
#             if partial_ratio >= 90:
#                 if partial_ratios_list == []:
#                     partial_ratios_list.append(partial_ratio)
#                     similar_words.append(word)
#                     similar_products.append(product)
#                 else:
#                     for i in range(len(partial_ratios_list)):
#                         if partial_ratio >= partial_ratios_list[i] and product not in similar_products:
#                             similar_words.append(word)
#                             similar_products.append(product)
#                             partial_ratios_list.append(partial_ratio)

#     customer_sentences = []
#     for i in range(len(similar_words)):
#         for j in range(len(tokens)):
#             if tokens[j] == similar_words[i]:
#                 old_keyword_token = tokens[j]
#                 tokens[j] = similar_products[i]
#                 new_customer_text = ''
#                 for h in tokens:
#                     new_customer_text += h+' '
#                 customer_sentences.append(new_customer_text)
#                 tokens[j] = old_keyword_token

#     if similar_words == []:
#         print("\n \nWe could not find a keyword in your text")

#     if customer_sentences == []:
#         customer_sentences.append(customer_text)
#     return customer_sentences


# #  Skin requirement detection Model


# def skinRequirement(message):
#     msgs = customer_says(message)
#     print(msgs, "ckeck message")
#     skincarelist = []
#     for msg in msgs:
#         print(msg, "msg")
#         if 'cleanser' in msg.lower() or 'clarifier' in msg.lower() or ('cleaning' in msg.lower() and 'agent' in msg.lower()) or 'disinfectant' in msg.lower() or 'refiner' in msg.lower() or 'refinery' in msg.lower() or 'soap' in msg.lower() or ('face' in msg.lower() and 'wash' in msg.lower()) or 'cleaner' in msg.lower() or 'clean' in msg.lower() or 'antibacterial' in msg.lower() or ('skin' in msg.lower() and 'clean' in msg.lower()) or ('makeup' in msg.lower() and 'remove' in msg.lower()) or ('clean' in msg.lower() and 'gel' in msg.lower()) or ('clean' in msg.lower() and 'milk' in msg.lower()) or ('clean' in msg.lower() and 'lotion' in msg.lower()) or ('clean' in msg.lower() and 'oil' in msg.lower()):
#             if 'SKINCARE - CLEANSER' not in skincarelist:
#                 pred_tag = 'SKINCARE - CLEANSER'
#                 skincarelist.append('SKINCARE - CLEANSER')
#         if 'toner' in msg.lower() or 'astringent' in msg.lower() or 'mattifier' in msg.lower() or ('clarifying' in msg.lower() and 'lotion' in msg.lower()) or 'tonic' in msg.lower():
#             if 'SKINCARE - TONER' not in skincarelist:
#                 pred_tag = 'SKINCARE - TONER'
#                 skincarelist.append('SKINCARE - TONER')
#         if 'exfoliant' in msg.lower() or 'scrub' in msg.lower():
#             if 'SKINCARE - Exfoliator' not in skincarelist:
#                 pred_tag = 'SKINCARE - Exfoliator'
#                 skincarelist.append('SKINCARE - Exfoliator')
#         if 'peel' in msg.lower():
#             if 'SKINCARE - Peels' not in skincarelist:
#                 pred_tag = "SKINCARE - Peels"
#                 skincarelist.append('SKINCARE - Peels')
#         if 'serum' in msg.lower() or 'booster' in msg.lower() or 'oil' in msg.lower() or 'moisturiser' in msg.lower() or 'complex' in msg.lower() or 'solution' in msg.lower() or 'elixir' in msg.lower() or 'potion' in msg.lower() or 'treatment' in msg.lower():
#             if 'SKINCARE - SERUM' not in skincarelist:
#                 pred_tag = 'SKINCARE - SERUM'
#                 skincarelist.append('SKINCARE - SERUM')
#         if ('day' in msg.lower() and 'moisturiser' in msg.lower()) or 'dry' in msg.lower() or 'lotion' in msg.lower() or 'cream' in msg.lower() or 'moisturiser' in msg.lower() or 'moisturizer' in msg.lower() or 'creme' in msg.lower() or ('night' in msg.lower() and 'moisturiser' in msg.lower()) or ('face' in msg.lower() and 'moisturiser' in msg.lower()) or ('face' in msg.lower() and 'moisturizer' in msg.lower()):
#             if 'SKINCARE - MOISTURISER' not in skincarelist:
#                 pred_tag = 'SKINCARE - MOISTURISER'
#                 skincarelist.append('SKINCARE - MOISTURISER')
#         if 'sunscreen' in msg.lower() or 'spf' in msg.lower() or 'sun lotion' in msg.lower() or 'skin protection' in msg.lower() or ('sun' in msg.lower() and 'factor' in msg.lower()) or 'protectant' in msg.lower() or 'spf 50' in msg.lower() or 'spf 10' in msg.lower() or 'spf 15' in msg.lower() or 'spf 30' in msg.lower() or 'spf 35' in msg.lower() or 'uv' in msg.lower() or 'uva' in msg.lower() or 'uvb' in msg.lower() or 'sunblock' in msg.lower() or 'sun tan' in msg.lower() or 'beach' in msg.lower():
#             if 'SKINCARE - SPF' not in skincarelist:
#                 pred_tag = 'SKINCARE - SPF'
#                 skincarelist.append('SKINCARE - SPF')
#         if ('eye' in msg.lower() and 'cream' in msg.lower()) or ('eye' in msg.lower() and 'serum' in msg.lower()) or 'anti wrinkle' in msg.lower() or 'dark circles' in msg.lower() or 'bags' in msg.lower() or 'repair' in msg.lower() or 'treatment' in msg.lower():
#             if 'SKINCARE - Eyes' not in skincarelist:
#                 pred_tags = 'SKINCARE - Eyes'
#                 skincarelist.append('SKINCARE - Eyes')
#         if 'mask' in msg.lower() or ('face' in msg.lower() and 'mask' in msg.lower()) or ('clay' in msg.lower() and 'mask' in msg.lower()) or ('sheet' in msg.lower() and 'mask' in msg.lower()) or 'face peel' in msg.lower() or 'masque' in msg.lower() or 'peel' in msg.lower() or 'overnight mask' in msg.lower() or 'repair mask' in msg.lower() or 'face pack' in msg.lower() or 'mud pack' in msg.lower() or 'treatment' in msg.lower():
#             if 'SKINCARE - MASK' not in skincarelist:
#                 pred_tag = 'SKINCARE - MASK'
#                 skincarelist.append('SKINCARE - MASK')
#         if ('night' in msg.lower() and 'cream' in msg.lower()) or ('lotion' in msg.lower() and 'night' in msg.lower()) or ('moist' in msg.lower() and 'night' in msg.lower()) or ('sleep' in msg.lower() and 'cream' in msg.lower()) or ('sleep' in msg.lower() and 'moist' in msg.lower()):
#             if 'SKINCARE - Night cream' not in skincarelist:
#                 pred_tag = 'SKINCARE - Night cream'
#                 skincarelist.append('SKINCARE - Night cream')
#         if ('face' in msg.lower() and 'spray' in msg.lower()) or ('spray' in msg.lower() and 'moist' in msg.lower()) or ('face' in msg.lower() and 'mist' in msg.lower()) or 'mist' in msg.lower():
#             if 'SKINCARE - Facial Mist' not in skincarelist:
#                 pred_tag = 'SKINCARE - Facial Mist'
#                 skincarelist.append('SKINCARE - Night cream')
#         if ('face' in msg.lower() and 'oil' in msg.lower()):
#             if 'SKINCARE - Face oil' not in skincarelist:
#                 pred_tag = 'SKINCARE - Face oil'
#                 skincarelist.append('SKINCARE - Face oil')
#         if 'lip' in msg.lower():
#             if 'SKINCARE - Lips' not in skincarelist:
#                 pred_tag = 'SKINCARE - Lips'
#                 skincarelist.append('SKINCARE - Lips')
#         if 'peel' in msg.lower():
#             if 'SKINCARE - Peels' not in skincarelist:
#                 pred_tag = 'SKINCARE - Peels'
#                 skincarelist.append('SKINCARE - Peels')
#         if 'anti-aging' in msg.lower():
#             if 'SKINCARE - anti-aging' not in skincarelist:
#                 pred_tag = 'SKINCARE - anti-aging'
#                 skincarelist.append('SKINCARE - anti-aging')
#         if 'vegan' in msg.lower() or ('cruelty' in msg.lower() and 'free' in msg.lower()):
#             if 'SKINCARE - Vegan skincare' not in skincarelist:
#                 pred_tag = 'SKINCARE - Vegan skincare'
#                 skincarelist.append('SKINCARE - Vegan skincare')

#     if len(skincarelist) > 0:
#         pass
#     else:
#         msg = message
#         ints = predict_class(msg, model, words_care, classes_care)
#         print(ints, "pred skin care ")
#         for int in ints:
#             pred_tag = int['intent']
#             prob_pred = int['probability']
#             if float(prob_pred) > 0.9:
#                 if pred_tag == 'SKINCARE - MOISTURISER' and prob_pred < 0.99:
#                     pass
#                 else:
#                     skincarelist.append(pred_tag)
#     return skincarelist


# #  Skin Type detection Model

# def skinType(message):
#     skintypelist = []
#     msg = message
#     if 'dry' in msg.lower():
#         pred_tag = 'SKINTYPE - DRY'
#         skintypelist.append('SKINTYPE - DRY')
#     if 'oily' in msg.lower() or 'oiliness' in msg.lower():
#         pred_tag = 'SKINTYPE - OILY'
#         skintypelist.append('SKINTYPE - OILY')
#     if 'normal' in msg.lower():
#         pred_tag = 'SKINTYPE - NORMAL'
#         skintypelist.append('SKINTYPE - NORMAL')
#     if 'sensitive' in msg.lower():
#         pred_tag = 'SKINTYPE - SENSITIVE'
#         skintypelist.append('SKINTYPE - SENSITIVE')
#     if 'combination' in msg.lower():
#         pred_tag = 'SKINTYPE - COMBINATION'
#         skintypelist.append('SKINTYPE - COMBINATION')

#     if len(skintypelist) > 0:
#         pass
#     else:
#         msg = message
#         ints = predict_class(msg, skintypemodel, words_t, classes_t)
#         print(ints, "pred skin type")
#         for int in ints:
#             pred_tag = int['intent']
#             prob_pred = int['probability']
#             if float(prob_pred) > 0.99:
#                 skintypelist.append(pred_tag)
#     return skintypelist


# #  Skin Concern detection Model

# def skinConcern(message):
#     msg1 = customer_says(message)
#     keyword_itch = re.compile('\\b.*itch.*\\b')
#     skinconcern = []
#     if msg1:
#         msgs = msg1
#     else:
#         msgs = [message]
#     print(msgs, "msgs")
#     for msg in msgs:
#         if "dry" in msg.lower() or "dehydrated" in msg.lower():
#             pred_tag = "SKINCONCERN - DRY & DEHYDRATED"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN - DRY & DEHYDRATED")
#         if "sensitivity" in msg.lower() or "irritation" in msg.lower() or re.search(keyword_itch, msg.lower()):
#             pred_tag = "SKINCONCERN -  Sensitivity & Irritation"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN -  Sensitivity & Irritation")
#         if "acne" in msg.lower() or "breakouts" in msg.lower() or "redhead" in msg.lower() or "blemishes" in msg.lower() or "pimples" in msg.lower():
#             pred_tag = "SKINCONCERN - Acne & Breakouts"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN - Acne & Breakouts")
#         if "uv" in msg.lower() or "damage" in msg.lower():
#             pred_tag = "SKINCONCERN - UV Damage"
#             qid = 6
#             skinconcern.append("SKINCONCERN - UV Damage")
#         if "puffiness" in msg.lower() or "redness" in msg.lower():
#             pred_tag = "SKINCONCERN - PUFFINESS AND REDNESS"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN - PUFFINESS AND REDNESS")
#         if "scarring" in msg.lower() or "pigmentation" in msg.lower():
#             pred_tag = "SKINCONCERN - SCARRING & PIGMENTATION"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN - SCARRING & PIGMENTATION")
#         if "loss of skin elasticity" in msg.lower():
#             pred_tag = "SKINCONCERN - Loss of skin elasticity"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN - Loss of skin elasticity")
#         if 'dull' in msg.lower():
#             pred_tag = "SKINCONCERN - DULLNESS"
#             qid = 6
#             skinconcern.append("SKINCONCERN - DULLNESS")
#         if "uneven skintone" in msg.lower():
#             pred_tag = "SKINCONCERN - UNEVEN"
#             qid = 6
#             skinconcern.append("SKINCONCERN - UNEVEN")
#         if "clogged pore" in msg.lower() or ('clogged' in msg.lower() and 'pore' in msg.lower()) or "heavy makeup" in msg.lower() or "smells bad" in msg.lower():
#             pred_tag = "SKINCONCERN - CLOGGED PORES"
#             qid = 6
#             skinconcern.append("SKINCONCERN - CLOGGED PORES")
#         if "fine lines" in msg.lower() or "wrinkles" in msg.lower() or "aging" in msg.lower() or "age" in msg.lower():
#             pred_tag = "SKINCONCERN - FINE LINES & WRINKLE"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN - FINE LINES & WRINKLE")
#         if 'blackhead' in msg.lower() or 'whitehead' in msg.lower():
#             pred_tag = "SKINCONCERN - BLACKHEADS & WHITEHEADS"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN - BLACKHEADS & WHITEHEADS")
#         if "excess oil" in msg.lower() or "oiliness" in msg.lower() or "oily" in msg.lower():
#             pred_tag = "SKINCONCERN - Excess Oil"
#             qid = 6
#             skinconcern.append("SKINCONCERN - Excess Oil")
#         if 'razor bumps' in msg.lower() or 'ingrown hairs' in msg.lower():
#             pred_tag = "SKINCONCERN - Razor bump & Ingrown Hairs"
#             qid = 6
#             skinconcern.append(
#                 "SKINCONCERN - Razor bump & Ingrown Hairs")
#         if ('no' in msg.lower() and ('concern' in msg.lower() or 'issues' in msg.lower())) or 'perfect' in msg.lower():
#             pred_tag = "SKINCONCERN - No Concerns"
#             qid = 6
#             skinconcern.append("SKINCONCERN - No Concerns")

#     print(skinconcern, "skinconcern")
#     if len(skinconcern) > 0:
#         pass
#     else:
#         msg = message
#         ints = predict_class(msg, skinconcernmodel,
#                              words_concern, classes_concern)
#         print(ints, "pred skin concern")
#         for int in ints:
#             pred_tag = int['intent']
#             prob_pred = int['probability']
#             if float(prob_pred) > 0.99:
#                 skinconcern.append(pred_tag)

#     skinconcern1 = skinconcern
#     skinconcern = []
#     for s in skinconcern1:
#         if s in skinconcern:
#             pass
#         else:
#             skinconcern.append(s)
#     return skinconcern


# #  Skin preferences

# def skinPreference(message):
#     ms = message.lower().replace('and ', ',').replace(
#         'or ', ',').replace('&', ',').replace('|', '').split(',')
#     skin_pref = []
#     for m in ms:
#         if 'silicone free' in m:
#             pred_tag = "SKINPREFERENCE - silicone free"
#         elif 'non gmo' in m:
#             pred_tag = "SKINPREFERENCE - non gmo"
#         elif 'paraben' in m:
#             pred_tag = "SKINPREFERENCE - parben"
#         elif 'steroid' in m:
#             pred_tag = "SKINPREFERENCE - steroid"
#         elif 'hypoallergenic' in m:
#             pred_tag = 'SKINPREFERENCE - hypoallergenic'
#         elif 'synthetic fragrance' in m:
#             pred_tag = 'SKINPREFERENCE - synthetic fragrance'
#         elif 'artificial dye' in m:
#             pred_tag = 'SKINPREFERENCE - artificial dyes'
#         elif 'alcohol' in m:
#             pred_tag = 'SKINPREFERENCE - alcohol free'
#         elif 'sulphate' in m:
#             pred_tag = 'SKINPREFERENCE - sulphate free'
#         elif 'vegan' in m:
#             pred_tag = 'SKINPREFERENCE - vegan'
#         elif 'vegetarian' in m:
#             pred_tag = 'SKINPREFERENCE - vegetarian'
#         elif 'organic' in m:
#             pred_tag = 'SKINPREFERENCE - organic'
#         elif 'cruelty' in m:
#             pred_tag = 'SKINPREFERENCE - cruelty free'
#         elif 'gluten' in m:
#             pred_tag = 'SKINPREFERENCE - gluten free'
#         elif 'phthalates' in m:
#             pred_tag = 'SKINPREFERENCE - phthalates free'
#         elif 'ethically sourced' in m:
#             pred_tag = 'SKINPREFERENCE - ethically sourced'
#         elif 'halal' in m:
#             pred_tag = 'SKINPREFERENCE - halal'
#         elif 'kosher' in m:
#             pred_tag = 'SKINPREFERENCE - kosher'
#         elif 'ethic' in m:
#             pred_tag = 'SKINPREFERENCE - ethically sourced'
#         elif 'pregnant' in m:
#             pred_tag = 'SKINPREFERENCE - pregnant'
#         elif 'menopause' in m:
#             pred_tag = 'SKINPREFERENCE - menopause'
#         elif 'no preference' in m or "don't care" in m or 'dont care' in m or 'no' in m:
#             pred_tag = 'SKINPREFERENCE - No Preference'
#         else:
#             pred_tag = 'SKINPREFERENCE - others'
#         skin_pref.append(pred_tag)
#     return skin_pref


# class recievereplyView(APIView):
#     serializer_class = recievereplySerializer
#     # permission_classes = (IsAuthenticated,)

#     def get_object(self, pk):
#         try:
#             return customer_chat.objects.filter(sessid=pk).last()
#         except customer_chat.DoesNotExist:
#             raise Http404

#     def get(self, request, pk,  format=None):
#         print(pk, "session id")
#         c = self.get_object(pk)
#         # print(c)
#         pred_tag = None
#         qid = None
#         res = None
#         # starting question
#         lastqid_q = customer_chat.objects.filter(
#             sessid=pk).filter(qid__isnull=False).last()
#         # print(lastqid_q, "first qid")

#         #  Run  skin requiremenr model to find pred_tag
#         #  Checking for the brand
#         if pred_tag is None or pred_tag == 'Reco - Name':
#             print("2 . brand search name")
#             brand_search_res = brand_search(
#                 c.question.lower().replace('skin', '').replace('care', ''))
#             if brand_search_res is not None:
#                 pred_tag = "Brand"
#                 qid = 1
#                 res = brand_search_res

#         # Checking for products and product category
#         if pred_tag is None:
#             print("1 . Checking for product category")
#             prod_search = product_category_search(c.question.lower())
#             if prod_search['result'] is not None:
#                 pred_tag = prod_search['pred_tag']
#                 if pred_tag == 'Product':
#                     qid = 1
#                     res = prod_search['result']
#                 else:
#                     qid = 1
#                     res = prod_search['result']

#         #  ckecking for ingredient
#         if pred_tag is None:
#             print("3. ingredient Search")
#             if 'what is' in c.question.lower() or 'definition' in c.question.lower() or 'explain' in c.question.lower() or 'meaning' in c.question.lower() or 'mean' in c.question.lower() or 'what does' in c.question.lower() or 'vocabulary' in c.question.lower() or 'explanation' in c.question.lower() or 'ingredients' in c.question.lower() or 'learn' in c.question.lower():
#                 ingredient_res = ingredient_search(c.question.lower())
#                 print(ingredient_res, "ingred")
#                 if ingredient_res:
#                     pred_tag = 'Ingredient'
#                     res = ingredient_res
#                     qid = 1

#         #  Checking for Customer Servive and Customer feedback
#         if pred_tag is None:
#             print("8. Customer Servive name", c.question)
#             cs_tag = CustomerService_detection(c.question)
#             # print(cs_tag, "test customer service")
#             if cs_tag is not None:
#                 pred_tag = cs_tag
#                 qid = 1
#                 if pred_tag == 'FEEDBACK':
#                     res = "We love hearing your feedback. Leave either a simple  or , or send an email direct to <span style='word-wrap: break-word;'><a href='mailto:hello@yuty.me'>hello@yuty.me</a></span> Thank you "
#                 else:
#                     name_q = customer_chat.objects.all().filter(
#                         sessid=pk).filter(pred_tag='Reco - Name')
#                     if name_q:
#                         contact_us = 'Hello ' + \
#                             str(name_q[0].question) + ". Have you checked out our  <a href='https://www.yuty.me/faq'>FAQs</a> and still can’t find what you’re looking for? To help you with your query, please contact <a href='mailto:customercare@yuty.me'>customercare@yuty.me</a>, Yuty <p> Would you like to continue, yes or no"
#                         res = contact_us
#                     else:
#                         contact_us = "Have you checked out our  <a href='https://www.yuty.me/faq'>FAQs</a> and still can’t find what you’re looking for? To help you with your query, please contact <a href='mailto:customercare@yuty.me'>customercare@yuty.me</a>, Yuty <p> Would you like to continue, yes or no"
#                         res = contact_us

#         #  starting the chat thread
#         if pred_tag is None:
#             if lastqid_q is None:
#                 # check if input is name
#                 print("6. Checking name")
#                 name = None
#                 msg = c.question.title()
#                 tokenized_text = word_tokenize(msg)
#                 classified_text = st.tag(tokenized_text)
#                 # print(tokenized_text, '\n', classified_text)

#                 for tag in classified_text:
#                     if tag[1] == 'PERSON':
#                         # print("Found tag", tag[1])
#                         pred_tag = "Reco - Name"
#                         name = tag[0]
#                 if pred_tag == "Reco - Name":
#                     res = "Nice to meet you, " + \
#                         str(name) + ".<br/>How can I help you with your skincare routine today?"
#                     qid = 1

#             else:
#                 print("7 . checking the recommendation")
#                 # print(1, "test", lastqid_q.pred_tag)
#                 if lastqid_q.pred_tag == 'Ingredient':
#                     yes_res = customer_chat.objects.filter(sessid=pk).filter(
#                         qid__isnull=False).filter(pred_tag__icontains='Reco').last()
#                     if yes_res is None:
#                         print("handle none")
#                         if c.question.lower() in ('no', 'nope'):
#                             pred_tag = 'Ingredient'
#                             res = 'Should you need any help with your skincare routine, please don’t hesitate to ask Yuty. #YutyKnows ✨'
#                         elif c.question.lower() in ('yes', 'yeah', 'ok'):
#                             res = "What's your name?"
#                             pred_tag = 'Name'
#                             qid = 2
#                     else:
#                         if yes_res.qid == 1:
#                             if c.question.lower() in ('no', 'nope'):
#                                 pred_tag = 'Ingredient'
#                                 res = 'Should you need any help with your skincare routine, please don’t hesitate to ask Yuty. #YutyKnows ✨'
#                             elif c.question.lower() in ('yes', 'yeah', 'ok'):

#                                 if yes_res is None:
#                                     res = "What's your name?"
#                                     pred_tag = 'Name'
#                                     qid = 2

#                                 else:
#                                     res = yes_res.response
#                                     if res.find("Nice to meet you") > -1:
#                                         res = "How can I help you with your skincare routine today?"
#                                     pred_tag = 'Ingredient'
#                         elif yes_res.qid == 3:
#                             if c.question.lower() in ('no', 'nope'):
#                                 pred_tag = 'Ingredient'
#                                 res = 'Should you need any help with your skincare routine, please don’t hesitate to ask Yuty. #YutyKnows ✨'
#                             else:
#                                 concern_pred = skinConcern(lastqid_q.question)
#                                 print(concern_pred, len(concern_pred))
#                                 if len(concern_pred) == 0:
#                                     res = "Our skin changes with time, taking care of and utilising products that address the skin-care issues that affect us irrespective of age. As a teenager your concerns may have included acne, scarring and pigmentation, whereas in your 50s concerns may include wrinkles, loss of skin elasticity and dryness. - what would you say are your main skin concerns?"
#                                     pred_tag = lastqid_q.pred_tag
#                                     qid = lastqid_q.qid
#                                 else:
#                                     qid = 4
#                                     pred_tag = "Reco - " + \
#                                         ', '.join(concern_pred)
#                                     res = "Right! So, I have understood your skin type, and concerns. Your skin is affected by where you live, so which city are you in?"

#                 elif lastqid_q.pred_tag == 'Name':
#                     print("6. Checking name")
#                     name = None
#                     msg = c.question.title()
#                     tokenized_text = word_tokenize(msg)
#                     classified_text = st.tag(tokenized_text)
#                     print(tokenized_text, '\n', classified_text)

#                     for tag in classified_text:
#                         # print(tag[0], '=======>',  tag[1])
#                         if tag[1] == 'PERSON':
#                             print("Found tag", tag[1])
#                             pred_tag = "Reco - Name"
#                             name = tag[0]
#                     if pred_tag == "Reco - Name":
#                         res = "Nice to meet you, " + \
#                             str(name) + ".<br/>How can I help you with your skincare routine today?"
#                         qid = 1
#                 elif lastqid_q.pred_tag == 'Brand':
#                     if lastqid_q.qid == 1:
#                         if c.question.lower() in ('no', 'nope'):
#                             pred_tag = 'Brand'
#                             res = 'Should you need any help with your skincare routine, please don’t hesitate to ask Yuty. #YutyKnows ✨'
#                         elif c.question.lower() in ('yes', 'yeah', 'ok'):
#                             yes_res = customer_chat.objects.filter(sessid=pk).filter(
#                                 qid__isnull=False).filter(pred_tag__icontains='Reco').last()
#                             if yes_res is None:
#                                 res = "What's your name?"
#                                 pred_tag = 'Name'
#                                 qid = 2
#                             else:
#                                 res = yes_res.response
#                                 if res.find("Nice to meet you") > -1:
#                                     res = "How can I help you with your skincare routine today?"
#                                 pred_tag = 'Brand'
#                 elif lastqid_q.pred_tag.find('Product') > -1:
#                     if lastqid_q.qid == 1:
#                         if c.question.lower() in ('no', 'nope'):
#                             pred_tag = 'Product'
#                             res = 'Should you need any help with your skincare routine, please don’t hesitate to ask Yuty. #YutyKnows ✨'
#                         elif c.question.lower() in ('yes', 'yeah', 'ok'):
#                             if lastqid_q.pred_tag == 'Product':
#                                 yes_res = customer_chat.objects.filter(sessid=pk).filter(
#                                     qid__isnull=False).filter(pred_tag__icontains='Reco').last()
#                                 if yes_res is None:
#                                     res = "What's your name?"
#                                     pred_tag = 'Name'
#                                     qid = 2
#                                 else:
#                                     res = yes_res.response
#                                     if res.find("Nice to meet you") > -1:
#                                         res = "How can I help you with your skincare routine today?"
#                                     pred_tag = 'Product'
#                             elif lastqid_q.pred_tag == 'Product Category':
#                                 res = "Okay, let's get started. What is your skintype? Would you say it's </br>DRY</br>OILY</br>BALANCED (Normal)</br>COMBINATION</br>SENSITIVE"
#                                 #  Run model on old response to predict requirement type
#                                 print(lastqid_q.question, "skin care check ")
#                                 req_pred = skinRequirement(lastqid_q.question)
#                                 print(req_pred, lastqid_q.question,
#                                       "skin care check result")
#                                 pred_tag = 'Reco - '+' , '.join(req_pred)
#                                 qid = 2
#                 elif lastqid_q.pred_tag.find('Customer Service') > -1:
#                     print("test 222")
#                     if lastqid_q.qid == 1:
#                         if c.question.lower() in ('no', 'nope'):
#                             pred_tag = 'Customer Service'
#                             res = 'Should you need any help with your skincare routine, please don’t hesitate to ask Yuty. #YutyKnows ✨'
#                         elif c.question.lower() in ('yes', 'yeah', 'ok'):
#                             yes_res = customer_chat.objects.filter(sessid=pk).filter(
#                                 qid__isnull=False).filter(pred_tag__icontains='Reco').last()
#                             if yes_res is None:
#                                 res = "What's your name?"
#                                 print("test 2221")
#                                 pred_tag = 'Name'
#                                 qid = 1
#                             else:
#                                 print("test 2222")
#                                 res = yes_res.response
#                                 if res.find("Nice to meet you") > -1:
#                                     res = "How can I help you with your skincare routine today?"
#                             if yes_res:
#                                 pred_tag = yes_res.pred_tag
#                                 qid = yes_res.qid
#                 elif lastqid_q.pred_tag.find("Reco") > -1:
#                     # print("in reco tree",
#                     #       lastqid_q.question, c.question)
#                     if lastqid_q.pred_tag.find("Name") > -1:
#                         # print("check skin care")
#                         req_pred = skinRequirement(c.question)
#                         # print(req_pred, len(req_pred))
#                         if len(req_pred) == 0:
#                             # Repeat question
#                             if lastqid_q:
#                                 num_confusion = customer_chat.objects.filter(sessid=pk).filter(Q(
#                                     pred_tag='Unsure') | Q(pred_tag='Spam')).filter(id__gte=lastqid_q.id)
#                             else:
#                                 num_confusion = []
#                             # print(len(num_confusion))
#                             if len(num_confusion) > 0:
#                                 res = "To help you find the right product, Yuty recommends taking the <a href='https://www.yuty.me/advisor'>YUTY Advisor ©</a> where Yuty will match you with conscious beauty products that fit you perfectly. #YutyKnows"

#                             else:
#                                 res = "How can I help you with your skincare routine today? For example you may be looking for a cleanser, serum, moisturiser or even SPF?"
#                             pred_tag = 'Unsure'
#                             qid = lastqid_q.qid
#                         else:
#                             qid = 2
#                             pred_tag = "Reco - " + \
#                                 ', '.join(req_pred)
#                             res = "Okay, let's get started. What is your skintype? Would you say it's </br>DRY</br>OILY</br>BALANCED (Normal)</br>COMBINATION</br>SENSITIVE"
#                     elif lastqid_q.pred_tag.find("SKINCARE") > -1:
#                         # print("check skin type")
#                         type_pred = skinType(c.question)
#                         print(type_pred, len(type_pred))
#                         if len(type_pred) == 0:
#                             if lastqid_q:
#                                 num_confusion = customer_chat.objects.filter(sessid=pk).filter(
#                                     Q(pred_tag='Unsure') | Q(pred_tag='Spam')).filter(id__gte=lastqid_q.id)
#                             else:
#                                 num_confusion = []
#                             print(len(num_confusion))
#                             if len(num_confusion) > 0:
#                                 res = "To help you find the right product, Yuty recommends taking the <a href='https://www.yuty.me/advisor'>YUTY Advisor © </a> where Yuty will match you with conscious beauty products that fit you perfectly. #YutyKnows"
#                             else:
#                                 res = "Okay let Yuty help. How does your skin usually feel? Oily, dry, balanced ('normal'), combination, and sensitive are the five kinds of skin type. By knowing what type of skin you have, you can start making educated decisions about how to care for and protect your skin now and in the future.DRY - does it feel dry and tight, prone to fine lines, with a dull appearance?OILY - does it feel oily, or look shiny in the t-zone? Do you have large pores and frequent breakouts?BALANCED (Normal) - doesn't really feel either oily or dry, and generally feels fairly balanced?COMBINATION - Feels quite oily in the t-zone, cheeks and other areas are usually dry? SENSITIVE - Sensitive skin is frequently referred to as a skin type, although you might have oily, dry, or balanced sensitive skin. It may be red, feel like it's burning, itchy, or dry if you have sensitive skin.These symptoms might be linked to having skin that is more sensitive to external irritants such as dyes or fragrances, as well as the environment. You may have eczema or rosacea"
#                             pred_tag = lastqid_q.pred_tag
#                             qid = lastqid_q.qid
#                         else:
#                             qid = 3
#                             pred_tag = "Reco - " + \
#                                 ', '.join(type_pred)
#                             res = "Thank you - What are your main skincare concerns?"
#                     elif lastqid_q.pred_tag.find("SKINTYPE") > -1:
#                         print("check skin concern")
#                         concern_pred = skinConcern(c.question)
#                         print(concern_pred, len(concern_pred))
#                         if len(concern_pred) == 0:
#                             #  Check if the confusion is more than 1
#                             if lastqid_q:
#                                 num_confusion = customer_chat.objects.filter(sessid=pk).filter(
#                                     Q(pred_tag='Unsure') | Q(pred_tag='Spam')).filter(id__gte=lastqid_q.id)
#                             else:
#                                 num_confusion = []
#                             print(len(num_confusion))
#                             if len(num_confusion) > 0:
#                                 res = "To help you find the right product, Yuty recommends taking the <a href='https://www.yuty.me/advisor'>YUTY Advisor © </a> where Yuty will match you with conscious beauty products that fit you perfectly. #YutyKnows"
#                             else:
#                                 res = "Our skin changes with time, taking care of and utilising products that address the skin-care issues that affect us irrespective of age. As a teenager your concerns may have included acne, scarring and pigmentation, whereas in your 50s concerns may include wrinkles, loss of skin elasticity and dryness. - what would you say are your main skin concerns?"
#                                 pred_tag = 'Unsure'
#                             # qid = lastqid_q.qid
#                         else:
#                             qid = 4
#                             pred_tag = "Reco - " + \
#                                 ', '.join(concern_pred)
#                             res = "Right! So, I have understood your skin type, and concerns. Your skin is affected by where you live, so which city are you in?"
#                     elif lastqid_q.pred_tag.find("SKINCONCERN") > -1:
#                         print("Capturing city name")
#                         pred_tag = "Reco - City - " + c.question
#                         qid = 5
#                         res = "Finally, what are your personal skincare preferences? You may prefer products which are vegan, cruelty free or without parabens? Or you simply may have no preference at all."
#                     elif lastqid_q.pred_tag.find("City") > -1:
#                         print("detecting prferences ")
#                         pref_pred = skinPreference(c.question)
#                         pred_tag = "Reco - Preference - "+', '.join(pref_pred)
#                         qid = 6
#                         res = "It’s been lovely getting to know you, Yuty will analyse and find products that will work for you."

#                 else:
#                     pass

#         print(pred_tag, qid, "spam test")

#         #  Spam , Thanks , Sorry , Good Bye etc .
#         if (pred_tag is None or pred_tag == 'Unsure') and lastqid_q is not None:
#             print("9. Spam/thanks/Sorry/GoodBye  Detection ")
#             num_confusion = customer_chat.objects.filter(sessid=pk).filter(
#                 Q(pred_tag='Unsure') | Q(pred_tag='Spam')).filter(id__gte=lastqid_q.id)
#             print(len(num_confusion), "number of confusion")
#             if len(num_confusion) > 1:
#                 res = "To help you find the right product, Yuty recommends taking the <a href='https://www.yuty.me/advisor'>YUTY Advisor © </a> where Yuty will match you with conscious beauty products that fit you perfectly. #YutyKnows"
#                 pred_tag = "Spam"
#             else:
#                 if lastqid_q.pred_tag == 'Unsure':
#                     res = "To help you find the right product, Yuty recommends taking the <a href='https://www.yuty.me/advisor'>YUTY Advisor © </a> where Yuty will match you with conscious beauty products that fit you perfectly. #YutyKnows"
#                     pred_tag = "Spam"
#                 else:
#                     spam_tag = regularExpression_detection(c.question)
#                     print("check spam test ", spam_tag)
#                     if spam_tag:
#                         if spam_tag.lower() == 'spam':
#                             res = "<span style='font-size:25px;'>&#x1f64a;</span>"
#                             pred_tag = "Spam"

#         if pred_tag is None:
#             #  Repeat previous question
#             if lastqid_q:
#                 num_confusion = customer_chat.objects.filter(sessid=pk).filter(
#                     Q(pred_tag='Unsure') | Q(pred_tag='Spam')).filter(id__gte=lastqid_q.id)
#             else:
#                 num_confusion = []
#             print(len(num_confusion), "number of confusion")
#             if len(num_confusion) > 0:
#                 res = "To help you find the right product, Yuty recommends taking the <a href='https://www.yuty.me/advisor'>YUTY Advisor © </a> where Yuty will match you with conscious beauty products that fit you perfectly. #YutyKnows"
#             else:
#                 if lastqid_q is None:
#                     res = "Hello 👋🏾, I'm YUTY, your AI beauty advisor, what's your name?"
#                     pred_tag = 'Unsure'
#                     # qid = None
#                 else:
#                     #  Check if the customer is failed to understand twice

#                     if lastqid_q.pred_tag.find('Reco') > -1 and lastqid_q.qid == 1:
#                         res = "How can I help you with your skincare routine today? For example you may be looking for a cleanser, serum, moisturiser or even SPF?"
#                     else:
#                         res = lastqid_q.response
#                     pred_tag = 'Unsure'
#                     # qid = lastqid_q.qid

#         print(c.question, pred_tag, qid, res)
#         customer_chat.objects.filter(id=c.id).update(
#             response=res, pred_tag=pred_tag, qid=qid)
#         print(res, pred_tag, "final output")
#         api_data = {
#             'text_mssg': res,
#             'pred_tag': pred_tag
#         }
#         return Response(api_data)


# # Chatbot recommendation


# class chatRecommendationView(APIView):
#     serializer_class = chatRecommendationSerializer
#     # permission_classes = (IsAuthenticated,)

#     def get_object(self, pk):
#         try:
#             return customer_chat.objects.filter(sessid=pk)
#         except customer_chat.DoesNotExist:
#             raise Http404

#     def get(self, request, pk,  format=None):
#         # s0 = datetime.now()
#         # try:
#         # EXTRACT REQUIREMENTS CAPTURED
#         data = {}
#         all_resp = customer_chat.objects.all().filter(sessid=pk).filter(
#             pred_tag__icontains='Reco').filter(qid__isnull=False)
#         reqs = {}
#         if all_resp:
#             skin_pref = []
#             for resp in all_resp:
#                 print(resp.pred_tag)
#                 if resp.pred_tag.find('SKINCARE') > -1:
#                     reqs['Skincare'] = resp.pred_tag.replace(
#                         'Reco - SKINCARE - ', "")
#                 elif resp.pred_tag.find('SKINTYPE') > -1:
#                     reqs['Skintype'] = resp.pred_tag.replace(
#                         "Reco - SKINTYPE - ", "")
#                 elif resp.pred_tag.find('SKINCONCERN') > -1:
#                     reqs['Skinconcern'] = resp.pred_tag.replace(
#                         "Reco - SKINCONCERN - ", "")
#                 elif resp.pred_tag.find("City") > -1:
#                     reqs['city'] = resp.pred_tag.replace("Reco - City -", "")
#                 elif resp.pred_tag.find('SKINPREFERENCE') > -1:
#                     skin_pref.append(resp.pred_tag.replace(
#                         "Reco - Preference - SKINPREFERENCE - ", ""))
#                 else:
#                     pass

#             reqs['skinpreference'] = skin_pref
#         skincare_params = []
#         print(reqs, "chat test ")
#         all_skincare = reqs['Skincare'].split(',')
#         for s in all_skincare:
#             skincare_params.append(s.lower().replace("skincare - ", ""))
#         # create recommendation :
#         all_skintype = reqs['Skintype'].split(',')
#         skintype_parameter = []
#         print(1)

#         for t in all_skintype:
#             skintype_parameter.append(t.lower().replace("skintype - ", ""))
#         skintype_parameter.append("all skin")
#         skinconcern_parameter = []
#         all_skinconcern = reqs['Skinconcern'].lower().replace(
#             "skinconcern - ", "").split(',')
#         print(2)
#         for c in all_skinconcern:
#             c1 = c.replace("skinconcern - ", "").replace("&",
#                                                          "~").replace("-", "~").split('~')
#             for c2 in c1:
#                 skinconcern_parameter.append(c2.lower())
#         all_skinpreference = reqs['skinpreference']
#         skinpreference_params = []
#         print(3)
#         for p in all_skinpreference:
#             if p.lower().replace("skinpreference - ", "") in ('no preference', 'others'):
#                 pass
#             else:
#                 skinpreference_params.append(
#                     p.lower().replace("skinpreference - ", ""))
#         products = []
#         print(4, skincare_params)
#         skincare_q = reduce(operator.or_, (Q(
#             product_category__icontains=item.lower()) for item in skincare_params))
#         print(4.2)
#         skintype_q = reduce(operator.or_, (Q(description__icontains=item) | Q(
#             good_to_know__icontains=item) for item in skintype_parameter))
#         skinconcern_q = reduce(operator.or_, (Q(description__icontains=item) | Q(
#             good_to_know__icontains=item) for item in skinconcern_parameter))
#         print(4.1, skinpreference_params)
#         if len(skinpreference_params) > 0:
#             skinpreference_q = reduce(operator.or_, (Q(description__icontains=item) | Q(
#                 good_to_know__icontains=item) for item in skinpreference_params))
#             productq = ApiAllProductUniq.objects.filter(skincare_q).filter(
#                 skintype_q).filter(skinconcern_q).filter(skinpreference_q)

#             print(len(productq), "level1")
#             if len(productq) < 2:
#                 productq = ApiAllProductUniq.objects.filter(
#                     skincare_q).filter(Q(skintype_q | skinconcern_q)).filter(Q(skinpreference_q | skinconcern_q)).filter(Q(skinpreference_q | skintype_q))
#                 print(len(productq), "level2")
#                 if len(productq) < 2:
#                     productq = ApiAllProductUniq.objects.filter(
#                         Q(skintype_q | skinconcern_q | skinpreference_q)).filter(skincare_q)
#                     print(len(productq), "level3")
#         else:
#             productq = ApiAllProductUniq.objects.filter(skincare_q).filter(
#                 skintype_q).filter(skinconcern_q)

#             print(len(productq), "level1")
#             if len(productq) < 2:
#                 productq = ApiAllProductUniq.objects.filter(
#                     skincare_q).filter(Q(skintype_q | skinconcern_q))
#                 print(len(productq), "level2")

#         print(len(productq), "numbert of products")
#         unq_prd = []
#         for p in productq:
#             print(p.product_id, p.product_name)
#             products.append({'product_id': p.product_id, 'product_name': p.product_name, 'product_category': p.product_category,
#                              'description': p.description, 'good_to_know': p.good_to_know, 'slug': p.slug})

#         data['requirements'] = reqs
#         data['recommendations'] = products[0:3]
#         # excpt:
#         #     data = {'error': 'No recommendation could be generated . '}
#         # print("Time to create user Cahtbot Recommendation: {} in secs.".format(
#         #     datetime.now() - s0))
#         return Response(data)


# #  Get Api to fetch the questions from id

class QuizQuestionView(APIView):
    # serializer_class = QuizQuestionSerializer
    # permission_classes = (IsAuthenticated,)
    """
    Retrieve, update or delete a snippet instance.
    """
    pass

    def get_object(self, pk):
        try:
            return SpreeQuestions.objects.get(slug=pk)
        except SpreeQuestions.DoesNotExist:
            raise Http404

    def get(self, request, pk,  format=None):
        qquestion = self.get_object(pk)
        serializer = QuizQuestionSerializer(qquestion)
        return Response(serializer.data)


class AnswerOptionView(APIView):
    # serializer_class = QuizQuestionSerializer
    # permission_classes = (IsAuthenticated,)
    """
    Retrieve, update or delete a snippet instance.
    """

    def get_object(self, pk):
        try:
            return SpreeAnswerOptions.objects.filter(spree_question_id=pk)
        except SpreeAnswerOptions.DoesNotExist:
            raise Http404

    def get(self, request, pk,  format=None):
        ans_options = self.get_object(pk)
        serializer = AnswerOptionSerializer(ans_options, many=True)
        return Response(serializer.data)


# # Create a post api to save input provided and important ingredients .


class customerResponseQuizView(APIView):

    def get(self, request, format=None):
        custresponse = customerquizresponse.objects.all()
        serializer = customerResponseQuizSerializers(custresponse, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        print(request.data, "dhdhdh")
        serializer = customerResponseQuizSerializers(data=request.data)
        print(serializer)
        if serializer.is_valid():
            print(request.data, "kfkfkfkkf")
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# #  Create Api for important ingrients

class ingredientView(APIView):
    """
    Retrieve, update or delete a snippet instance.
    """
    pass

    def get_object(self, pk):
        try:
            return ApiQuizquestionsNew.objects.get(choice_code=pk)
        except ApiQuizquestionsNew.DoesNotExist:
            raise Http404

    def get(self, request, pk,  format=None):
        qquestion = self.get_object(pk)
        serializer = ingredientSerializer(qquestion)
        return Response(serializer.data)


class IngredientRecommendationViewSkinCare(APIView):
    serializer_class = IngredientRecommendationSerializer

    def get_object(self, pk):
        try:
            return customerquizresponse.objects.filter(sess_id=pk)
        except customerquizresponse.DoesNotExist:
            raise Http404

    def cust_resp(self,  pk):
        customer_response = self.get_object(pk)
        all_resp = {}
        all_func = []
        non_reco_func = []
        for resp in customer_response:
            print(resp.slug, resp.option_text, resp.spree_option_id)
            # Pull tags for the optionss
            tags = ApiQuizquestionsNew.objects.values(
                'taxons', 'recommendation', 'non_recommendation').filter(choice_code=resp.spree_option_id)
            all_resp[resp.slug] = tags[0]['taxons']
            if tags[0]['recommendation']:
                for func in tags[0]['recommendation'].split(','):
                    all_func.append(func)
            if tags[0]['non_recommendation']:
                for nonfunc in tags[0]['non_recommendation'].split(','):
                    non_reco_func.append(nonfunc)
            print(tags)

        return all_resp, all_func, non_reco_func

    def get(self, request,  pk, format=None):
        data = {}
        all_resp, all_func, non_reco_func = self.cust_resp(pk)
        print(all_resp, all_func, non_reco_func, "resp")
        #  All non recomendable products
        nonreco_product = []
        reco_product = []

        if 'SQ7' in all_resp.keys():
            non_reco_func.append(all_choices_ls['SQ7'])
        if len(non_reco_func) > 0:
            noningred_filter = reduce(operator.or_, (Q(
                ingredients__icontains=item) for item in non_reco_func))
            nontaxon_product = reduce(operator.or_, (Q(
                all_info__icontains=item) for item in non_reco_func))
            nondesc_product = reduce(operator.or_, (Q(
                description__icontains=item) for item in non_reco_func))
            non_goodtoknow_product = reduce(operator.or_, (Q(
                good_to_know__icontains=item) for item in non_reco_func))
            non_product_q = ApiAllProductUniq.objects.all().filter(Q(noningred_filter) | Q(
                nontaxon_product) | Q(nondesc_product) | Q(non_goodtoknow_product)).order_by('product_id')

            for nprod in non_product_q:
                nonreco_product.append(nprod.product_id)

        my_dict = {i: all_func.count(i) for i in all_func}
        # create list of tables
        function_list = {'solvent': 0, 'sunscreen': 0, 'soothing': 0, 'cell_communicating_ingredient': 0,  'perfuming':  0,  'emollient': 0, 'emulsion_stabilising': 0,  'moisturizer_humectant': 0, 'buffering': 0, 'skin_identical_ingredient': 0, 'skin_brightening': 0, 'viscosity_controlling': 0,
                         'absorbent_mattifier': 0, 'deodorant': 0, 'antimicrobial_antibacterial': 0, 'astringent': 0, 'antioxidant': 0, 'surfactant_cleansing': 0, 'abrasive_scrub': 0, 'colorant': 0, 'emulsifying': 0, 'anti_acne': 0, 'preservative': 0, 'exfoliant': 0, 'chelating': 0}
        func_req = []
        reco = []
        for key in my_dict:
            if key.replace('-', '_').lower().strip() in my_dict.keys():
                function_list[key.replace(
                    '-', '_').lower().strip()] = my_dict[key]*100/sum(my_dict.values())
        all_func = set(list(all_func))

        prod_list_q = SpreeTaxons.objects.values('name').filter(parent_id=288)
        prod_list = []
        prod_unq = []
        for prod in prod_list_q:
            print(prod)
            prod_list.append(prod['name'])
            # hqprod = reduce(operator.or_, (Q(all_info__icontains=item)
            #                 for item in prod))

            prd_vec = ApiProductvectorNew.objects.values('prod_id', 'prod_name', 'brand', 'key_ingredient', 'all_info',  'solvent', 'sunscreen', 'soothing', 'cell_communicating_ingredient', 'perfuming', 'emollient', 'emulsion_stabilising', 'moisturizer_humectant', 'buffering', 'skin_identical_ingredient', 'skin_brightening', 'viscosity_controlling', 'absorbent_mattifier', 'deodorant',
                                                         'antimicrobial_antibacterial', 'astringent', 'antioxidant', 'surfactant_cleansing', 'abrasive_scrub', 'colorant', 'emulsifying', 'anti_acne', 'preservative', 'exfoliant', 'chelating').exclude(prod_id__in=nonreco_product).exclude(product_category__icontains='body').filter(all_info__icontains=prod['name'])
            df = pd.DataFrame(list(prd_vec))
            score_list = []
            brand_unq = []
            print("the number of product return step 1 ", prod,  df.shape[0])
            for i in range(len(df)):
                scr = 0
                scr_ingred = (df.loc[[i]]['solvent'].values[0] * function_list['solvent']) + (df.loc[[i]]['sunscreen'].values[0] * function_list['sunscreen']) + (df.loc[[i]]['soothing'].values[0] * function_list['soothing']) + (df.loc[[i]]['cell_communicating_ingredient'].values[0] * function_list['cell_communicating_ingredient']) + (df.loc[[i]]['perfuming'].values[0] * function_list['perfuming']) + (df.loc[[i]]['emollient'].values[0] * function_list['emollient']) + (df.loc[[i]]['emulsion_stabilising'].values[0] * function_list['emulsion_stabilising'])+(df.loc[[i]]['moisturizer_humectant'].values[0] * function_list['moisturizer_humectant'])+(df.loc[[i]]['buffering'].values[0] * function_list['buffering'])+(df.loc[[i]]['skin_identical_ingredient'].values[0] * function_list['skin_identical_ingredient'])+(df.loc[[i]]['skin_brightening'].values[0] * function_list['skin_brightening'])+(df.loc[[i]]['viscosity_controlling'].values[0] * function_list['viscosity_controlling'])+(
                    df.loc[[i]]['absorbent_mattifier'].values[0] * function_list['absorbent_mattifier'])+(df.loc[[i]]['deodorant'].values[0] * function_list['deodorant'])+(df.loc[[i]]['antimicrobial_antibacterial'].values[0] * function_list['antimicrobial_antibacterial'])+(df.loc[[i]]['astringent'].values[0] * function_list['astringent'])+(df.loc[[i]]['antioxidant'].values[0] * function_list['antioxidant'])+(df.loc[[i]]['surfactant_cleansing'].values[0] * function_list['surfactant_cleansing'])+(df.loc[[i]]['abrasive_scrub'].values[0] * function_list['abrasive_scrub'])+(df.loc[[i]]['colorant'].values[0] * function_list['colorant'])+(df.loc[[i]]['emulsifying'].values[0] * function_list['emulsifying'])+(df.loc[[i]]['anti_acne'].values[0] * function_list['anti_acne'])+(df.loc[[i]]['preservative'].values[0] * function_list['preservative'])+(df.loc[[i]]['exfoliant'].values[0] * function_list['exfoliant'])+(df.loc[[i]]['chelating'].values[0] * function_list['chelating'])
                if df.loc[[i]]['prod_id'].values[0] in prod_unq:
                    pass
                else:
                    score_list.append({'product_id': df.loc[[i]]['prod_id'].values[0], 'product_name': df.loc[[
                                      i]]['prod_name'].values[0],  'product_category': '/ '.join(prod).title(),  'brand': df.loc[[i]]['brand'].values[0], 'score': scr_ingred})

                    prod_unq.append(df.loc[[i]]['prod_id'].values[0])
            scr_list1 = sorted(
                score_list, key=lambda i: i['score'], reverse=True)

            brand_unq1 = []
            n = 0
            for s in scr_list1:
                if n < 3:
                    if s['brand'] in brand_unq1:
                        pass
                    else:
                        reco_product.append(s)
                        brand_unq1.append(s['brand'])
        # ApiProductvectorNew SpreeTaxons  select name  from spree_taxons where parent_id = 288

        print(function_list, prod['name'], len(prd_vec), "nonreco_product")
        data['reco'] = reco_product
        data['req'] = function_list

        return Response(data)


def product_esearch(product_name=""):
    client = Elasticsearch()
    q = esQ("multi_match", query=product_name, fields=[
        "brand_name", "description", "product_name"], fuzziness="AUTO")
    s = Search(using=client, index="products").query(
        q)[0:50]  # This will search and return up to 2 results
    response = s.execute()
    # print('%d hits found.' % response.hits.total['value'])
    search = get_results(response)
    return search


# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>')


def cleanhtml(raw_html):
    html = raw_html.replace('&nbsp;', ' ')
    cleantext = re.sub(CLEANR, '', html)
    return cleantext


def get_results(response):
    results = []
    for hit in response:
        print(hit)
        result_tuple = {'product_name': hit.product_name,
                        'product_id': hit.product_id,
                        'brand_name': hit.brand_name,
                        'description': cleanhtml(hit.description)
                        }
        results.append(result_tuple)
    return results


if __name__ == '__main__':
    print("YUTYBAZAR product details:\n", esearch(product_name="red lipstick"))


# Add product search api

class product_es_searchView(APIView):
    """
        Add a rest API pipeline for search query
    """

    def get(self, request, pk,  format=None):

        #  Call the elastic search server
        query = pk.replace(' ', '%20')
        results = product_esearch(product_name=pk)
        return Response(results)


# # Pull all product name for options in seartch as you type

# class allProductViews(APIView):

#     def get(self, request, format=None):
#         products = SpreeProducts.objects.values('name').distinct()
#         serializer = allProductSerializer(products, many=True)
#         return Response(serializer.data)


# #  Upload image to server

class FoundationImageUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):

        file_serializer = FoundationImageSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            print("is valid")

            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            print("not valid")
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


FACE = frozenset([

    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)
])


CONNECTIONS_FOREHEAD = frozenset([
    # Forehead.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10),
    (54, 284)
])
# #  Chin
CONNECTIONS_CHIN = frozenset([
    (83, 18),
    (18, 313),
    (313, 406),
    (406, 335),
    (335, 422),
    (422, 430),
    (430, 394),
    (394, 395),
    (395, 369),
    (369, 396),
    (396, 175),
    (175, 148),
    (148, 176),
    (176, 150),
    (150, 136),
    (136, 210),
    (210, 43),
    (43, 83)
])
#  Left Cheek
CONNECTIONS_LEFT_CHEEKS = frozenset([
    (116, 50),
    (50, 101),
    (101, 120),
    (120, 47),
    (47, 126),
    (126, 209),
    (209, 203),
    (203,  165),
    (165, 214),
    (214, 135),
    (135, 172),
    (172, 177),

    (177, 137),
    (137, 116)
])

#     #  Right Cheek
CONNECTIONS_RIGHT_CHEEK = frozenset([
    (264, 372),
    (372, 346),
    (346, 347),
    (347, 330),
    (330, 329),
    (329, 277),
    (277, 371),
    (371, 423),
    (423, 393),
    (393, 322),
    (322, 436),
    (436, 434),
    (434, 416),
    (416, 401),
    (401, 366),
    (366, 447),

    (447, 264)

])

CONNECTIONS_NOSE = frozenset([
    #  Nose
    (55, 8),
    (8, 285),
    (285, 417),
    (417, 351),
    (351, 419),
    (419, 248),
    (248, 275),
    (275, 1),
    (1, 45),
    (45, 51),
    (51, 188),
    # ( 188 ,189),
    (188, 193),
    (193, 55)

])


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


@ jit(nopython=True)
def background_removal(img61):
    # file = settings.MEDIA_ROOT+'/logo/line_imagenew.png'
    # img61 = cv2.imread(file)
    # img61 = self.skinExtraction(pk)
    # cv2.imwrite(settings.MEDIA_ROOT+'/logo/blacked1.png', img61)
    counter = 0
    green_dots = []
    for y in range(img61.shape[0]):
        for x in range(img61.shape[1]):
            if img61[y, x][0] == 0 and img61[y, x][1] == 0 and img61[y, x][2] == 0:
                counter = counter + 1
                green_dots.append([y, x])
    print(counter, len(green_dots))
    # # create a list with x min , x max and y coordinates of green points
    img6 = img61.copy()
    # cv2.imwrite(settings.MEDIA_ROOT+'/logo/blacked2.png', img6)
    green_area = []
    for y in range(img6.shape[0]):
        greenx_min = 10000
        greenx_max = 0
        for x in range(img6.shape[1]):
            for i in range(len(green_dots)):
                # print(i , "length of point")
                if green_dots[i][0] == y:
                    for green_x in green_dots:
                        if green_x[0] == y and green_x[1] < greenx_min:
                            greenx_min = green_x[1]
                        if green_x[0] == y and green_x[1] > greenx_max:
                            greenx_max = green_x[1]
                    if greenx_max != 10000 and greenx_max != 0:
                        # print('greenx min is ', greenx_min, 'greenx max is ', greenx_max, 'and y is ', y)
                        green_area.append([greenx_min, greenx_max, y])
    # left and right side make it black
    print("making it black ", len(green_area))
    for y in range(img6.shape[0]):
        # if y % 10 == 0:
        print((y * 100) / img6.shape[0])
        for x in range(img6.shape[1]):
            found = False
            for i in range(len(green_area)):
                # print(green_area[i][2] ,  y )
                if green_area[i][2] == y:
                    found = True
                    x_min = green_area[i][0]
                    x_max = green_area[i][1]
                    # print(i , x_min , x_max , x  )
                    if x < x_min:
                        # print("outside left side")
                        img6[y][x][0] = 0
                        img6[y][x][1] = 0
                        img6[y][x][2] = 0
                    elif x > x_max:
                        # print("outside right side")
                        img6[y][x][0] = 0
                        img6[y][x][1] = 0
                        img6[y][x][2] = 0
                    else:
                        # if img6[y][x][0] == 0 and img6[y][x][1] == 0 and img6[y][x][2] == 0:
                        #     pass
                        # else:
                        #     print(img6[y][x][0], img6[y]
                        #           [x][1], img6[y][x][2])
                        pass
                else:
                    pass
                    # print("outside right side")
            if found:
                pass
            else:
                img6[y][x][0] = 0
                img6[y][x][1] = 0
                img6[y][x][2] = 0
    # cv2.imwrite(settings.MEDIA_ROOT+'/logo/blacked.png', img6)
    return img6


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


# GET COLOUR INFORMATION

def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has been applied, remove the black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances

    # totalOccurance = sum(occurance_counter.values())
    val = occurance_counter.values()
    totalOccurance = sum(list(val))
    print(occurance_counter,
          "nddj", sum(val))
    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()
        color = [int(c) for c in color]

        # Get the percentage of each color
        print(x, totalOccurance, 'check1')
        color_percentage = (x[1]/totalOccurance)

        # make the dictionary of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=3, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert Image into RGB Colours Space in the case you use imutils.url_to_image the image will be saved in BGR format
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    newimg = []
    for i in img:
        if i[0] == 0 and i[1] == 0 and i[2] == 0:
            pass
        else:
            newimg.append(i)

    print(len(img), len(newimg), ">>>>>>>>>>>>>>>>>")
    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(newimg)

    # Get Color Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x["color"])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


# def pretty_print_data(color_info):
#     for x in color_info:
#         print(pprint.pformat(x))
#         print()


class ExtractSkinViewFoundation(generics.RetrieveAPIView):
    # serializer_class = ExtractSkinSerializer

    def get_object(self, pk):
        try:
            return foundation_image.objects.filter(imageid=pk).last()
        except foundation_image.DoesNotExist:
            raise Http404

    def selfieExtraction(self, pk):
        image = self.get_object(pk)
        print(settings.MEDIA_ROOT+'/'+str(image.imagename), "imagename")
        filename = settings.MEDIA_ROOT+'/'+str(image.imagename)
        f_landmark = []
        IMAGE_FILES = [filename]
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            for idx, file in enumerate(IMAGE_FILES):
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                annotated_image = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                    print("face_landmarks.landmarks",
                          len(face_landmarks.landmark))
                    i = 0
                    for pointmarks in face_landmarks.landmark:
                        print(pointmarks)
                        height, width, channels = image.shape
                        x = np.int0(pointmarks.x * width)
                        y = np.int0(pointmarks.y * height)
                        f_landmark.append({'index': i, 'co-ordinates': [x, y]})
                        i = i + 1
        # cv2.imwrite(settings.MEDIA_ROOT+'/logo/annotated.png', annotated_image)

        #  Draw line on the face in image
        line_image = image.copy()
        print(line_image.shape)
        for c in FACE:
            # print(c[0], type(c))
            i = 1
            for f in f_landmark:
                # print( i , f['index'] , c[0] , type(f['co-ordinates'][0]))
                if f['index'] == c[0]:
                    start = (f['co-ordinates'][0], f['co-ordinates'][1])
                    start_index = f['index']
                elif f['index'] == c[1]:
                    end = (f['co-ordinates'][0], f['co-ordinates'][1])
                    end_index = f['index']
                i = i + 1
            # print( start , end )
            line_image = cv2.line(line_image, start, end, (255, 0, 0), 3)
        cv2.imwrite(settings.MEDIA_ROOT+'/logo/line_image.png', line_image)
        latest = foundation_image(
            imageid=str(pk)+'_100', imagename='logo/line_image.png')
        latest.save()
        # Remove most of the background
        counter = 0
        green_dots = []
        xmin = 10000
        xmax = 0
        ymin = 10000
        ymax = 0
        for y in range(line_image.shape[0]):
            for x in range(line_image.shape[1]):
                if line_image[y, x][0] == 255 and line_image[y, x][1] == 0 and line_image[y, x][2] == 0:
                    counter = counter + 1
                    green_dots.append([y, x])
                    if y > ymax:
                        ymax = y
                    if x > xmax:
                        xmax = x

                    if y < ymin:
                        ymin = y
                    if x < xmin:
                        xmin = x

        #  Crop image
        crop_img = image[ymin: ymax, xmin: xmax]
        dim = (400, 400)
        resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(settings.MEDIA_ROOT+'/logo/cropped_img.png', resized)
        latest = foundation_image(
            imageid=str(pk)+'_101', imagename='logo/cropped_img.png')
        latest.save()
        return f_landmark

    def skinExtraction(self, pk):
        IMAGE_FILES = [settings.MEDIA_ROOT+'/logo/cropped_img.png']
        # filename = settings.MEDIA_ROOT+'/logo/cropped_img.png'
        # image = cv2.imread(filename)
        f_landmark = []
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            for idx, file in enumerate(IMAGE_FILES):
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                annotated_image = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    print("face_landmarks.landmarks",
                          len(face_landmarks.landmark))
                    i = 0
                    for pointmarks in face_landmarks.landmark:
                        print(pointmarks)
                        height, width, channels = image.shape
                        x = np.int0(pointmarks.x * width)
                        y = np.int0(pointmarks.y * height)
                        f_landmark.append({'index': i, 'co-ordinates': [x, y]})
                        i = i + 1

        #  Add the line
        # draw the line
        line_image = image.copy()
        print(line_image.shape)
        xmin = 100000
        ymin = 100000
        xmax = 0
        ymax = 0
        for c in CONNECTIONS_LEFT_CHEEKS:
            # print(c[0], type(c))
            i = 1
            for f in f_landmark:
                print(i, f['index'], c[0], (f['co-ordinates'][0]))
                if f['index'] == c[0]:
                    start = (f['co-ordinates'][0], f['co-ordinates'][1])
                    start_index = f['index']
                    if f['co-ordinates'][0] < xmin:
                        xmin = f['co-ordinates'][0]
                    if f['co-ordinates'][1] < ymin:
                        ymin = f['co-ordinates'][1]
                    if f['co-ordinates'][0] > xmax:
                        xmax = f['co-ordinates'][0]
                    if f['co-ordinates'][1] > ymax:
                        ymax = f['co-ordinates'][1]
                elif f['index'] == c[1]:
                    end = (f['co-ordinates'][0], f['co-ordinates'][1])
                    end_index = f['index']
                    if f['co-ordinates'][0] < xmin:
                        xmin = f['co-ordinates'][0]
                    if f['co-ordinates'][1] < ymin:
                        ymin = f['co-ordinates'][1]
                    if f['co-ordinates'][0] > xmax:
                        xmax = f['co-ordinates'][0]
                    if f['co-ordinates'][1] > ymax:
                        ymax = f['co-ordinates'][1]
                i = i + 1
                # print(start, end, 'start ')
            line_image = cv2.line(line_image, start, end, (0, 0, 0), 2)
        crop_img = line_image[np.int(ymin): np.int(
            ymax), np.int(xmin): np.int(xmax)]
        cv2.imwrite(settings.MEDIA_ROOT+'/logo/line_imagenew.png', crop_img)
        latest = foundation_image(
            imageid=str(pk)+'_102', imagename='logo/line_imagenew.png')
        latest.save()

        return crop_img

    # Blackout the undesired part

    def get(self, request, pk,  format=None):
        # data = []
        # Pull image name from database using id
        image = self.get_object(pk)
        # image = self.get_object(1)
        foundation_image.objects.filter(imageid='100').delete()
        foundation_image.objects.filter(imageid='101').delete()
        foundation_image.objects.filter(imageid='102').delete()
        foundation_image.objects.filter(imageid='103').delete()

        print(settings.MEDIA_ROOT+'/'+str(image.imagename), "imagename")
        f_landmarks = self.selfieExtraction(pk)
        # self.skinExtraction(pk)
        img61 = self.skinExtraction(pk)
        img_blacked = background_removal(img61)
        cv2.imwrite(settings.MEDIA_ROOT+'/logo/blacked3.png', img_blacked)
        latest = foundation_image(
            imageid=str(pk)+'_103', imagename='logo/blacked3.png')
        latest.save()
        color_info = extractDominantColor(
            img_blacked, number_of_colors=2, hasThresholding=False)
        print(color_info, type(color_info))
        return Response(color_info)


# #  Adding api to retrieve images


class ImageAPIViews(generics.RetrieveAPIView):
    renderer_classes = [PNGRenderer]

    def get(self, request, *args, **kwargs):

        print(self.kwargs['imageid'], "imageid")
        queryset = foundation_image.objects.filter(
            imageid=self.kwargs['imageid']).last().imagename
        data = queryset
        print(data, "<<<<<<<<<<<")
        return Response(data, content_type='image/png')

#  Add API for tryons liipsticks

#  POST image for try-ons . image will be sent and retuens with selected shades


# import dlib
predictor_loc = "/Users/macbookpro/demo/demo/media/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_loc)


class TryonImageUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):

        file_serializer = TryonsImageSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            print("is valid")

            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            print("not valid")
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def inter(lx, ly, k1='quadratic'):
    # evenly spaced values within given interval
    unew = np.arange(lx[0], lx[-1] + 1, 1)
    f2 = interp1d(lx, ly, kind=k1, fill_value="extrapolate")
    return f2, unew


class TryonsImageAPIViews(generics.RetrieveAPIView):
    renderer_classes = [PNGRenderer]

    def get(self, request, *args, **kwargs):

        print(self.kwargs['sess_id'], "imageid")
        queryset = tryons_image.objects.filter(
            sess_id=self.kwargs['sess_id']).last().output_image
        data = queryset
        print(data, "<<<<<<<<<<<")
        return Response(data, content_type='image/png')


class TryOnsImageLipstickView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):

        file_serializer = TryOnsSerializers(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            # print("is valid", request.POST['sess_id'])

            #  save in output
            imgObject = tryons_image.objects.filter(
                sess_id=request.POST['sess_id']).last()
            imgPath = settings.MEDIA_ROOT+'/'+str(imgObject.input_image)
            print(str(imgObject.input_image).split('/')[-1], "POST")
            # Apply tryons here
            img = cv2.imread(imgPath)
            # plt.imshow(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )
            imgcpy1 = img.copy()  # copy of image
            cv2.imwrite(settings.MEDIA_ROOT +
                        '/logo/imgcpy1.png', imgcpy1)
            for (x, y, w, h) in faces:
                cv2.rectangle(imgcpy1, (x, y), (x+w, y+h), (0, 255, 0), 2)

            outer_lip_x = []
            outer_lip_y = []
            # inner lip coordinates
            inner_lip_x = []
            inner_lip_y = []
            # mid of face
            mid_x = []
            mid_y = []
            # left cheek boundary
            cheek_left_x = []
            cheek_left_y = []
            # right cheek boundary
            cheek_right_x = []
            cheek_right_y = []

            # Load the detector
            detector = dlib.get_frontal_face_detector()
            # Load the predictor
            predictor = dlib.shape_predictor(
                "/Users/macbookpro/demo/demo/media/shape_predictor_68_face_landmarks.dat")

            # Use detector to find landmarks
            faces = detector(img)
            for face in faces:
                x1 = face.left()  # left point
                y1 = face.top()  # top point
                x2 = face.right()  # right point
                y2 = face.bottom()  # bottom point
                # Create landmark object
                landmarks = predictor(image=img, box=face)
                # Loop through all the points
                with open('points.txt', 'w') as writefile:
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        writefile.write(str(x)+" "+str(y)+"\n")

                        if n == 29:
                            mid_x.append(x)
                            mid_y.append(y)
                            cv2.circle(img=imgcpy1, center=(x, y), radius=3, color=(
                                100, 200, 10), thickness=-1)  # plot mid

                        if n in range(13, 16) or n in (35, 45):
                            cheek_right_x.append(x)
                            cheek_right_y.append(y)
                            cv2.circle(img=imgcpy1, center=(x, y), radius=3, color=(
                                0, 255, 255), thickness=-1)  # plot right cheek

                        if n in range(48, 60):
                            outer_lip_x.append(x)
                            outer_lip_y.append(y)
                            cv2.circle(img=imgcpy1, center=(x, y), radius=3, color=(
                                0, 0, 255), thickness=-1)  # plot outerlips
                        if n in range(60, 68):
                            inner_lip_x.append(x)
                            inner_lip_y.append(y)
                            cv2.circle(img=imgcpy1, center=(x, y), radius=3, color=(
                                255, 0, 0), thickness=-1)  # plot innerlips
            cv2.imwrite(settings.MEDIA_ROOT +
                        '/logo/imgcpy2.png', imgcpy1)
            # show the image
            outer_lip_x = np.array(outer_lip_x)
            outer_lip_y = np.array(outer_lip_y)
            inner_lip_x = np.array(inner_lip_x)
            inner_lip_y = np.array(inner_lip_y)
            cheek_left_x = np.array(cheek_left_x)
            cheek_left_y = np.array(cheek_left_y)
            cheek_right_x = np.array(cheek_right_x)
            cheek_right_y = np.array(cheek_right_y)

            # b1, g1, r1 = int(request.POST['b_value']), int(
            #     request.POST['g_value']), int(request.POST['r_value'])
            # #    "flamenco_red": (42., 31., 192.),
            # b1, g1, r1 = (177, 55, 75)
            rgb_value = str(request.POST['rgb_value'].replace('rgb', '').replace(
                '(', '').replace(')', '')).split(',')
            b1, g1, r1 = (int(rgb_value[2]), int(
                rgb_value[1]), int(rgb_value[0]))

            outer_left_end = 4
            outer_right_end = 7
            inner_left_end = 3
            inner_right_end = 5
            print(outer_lip_x,
                  outer_lip_y)
            outer_upper_left = inter(outer_lip_x[:outer_left_end],
                                     outer_lip_y[:outer_left_end])
            outer_upper_right = inter(outer_lip_x[outer_left_end - 1:outer_right_end],
                                      outer_lip_y[outer_left_end - 1:outer_right_end])

            x1 = [outer_lip_x[0]] + \
                outer_lip_x[outer_right_end - 1:][::-1].tolist()
            y1 = [outer_lip_y[0]] + \
                outer_lip_y[outer_right_end - 1:][::-1].tolist()
            outer_lip = inter(x1, y1, 'cubic')

            inner_upper_left = inter(
                inner_lip_x[:inner_left_end], inner_lip_y[:inner_left_end])
            inner_upper_right = inter(
                inner_lip_x[inner_left_end - 1:inner_right_end], inner_lip_y[inner_left_end - 1:inner_right_end])

            x2 = [inner_lip_x[0]] + \
                inner_lip_x[inner_right_end - 1:][::-1].tolist()
            y2 = [inner_lip_y[0]] + \
                inner_lip_y[inner_right_end - 1:][::-1].tolist()
            inner_lip = inter(x2, y2, 'cubic')

            x = []
            y = []

            def extension(a, b, i):
                a, b = np.round(a), np.round(b)
                x.extend(arange(a, b, 1, dtype=np.int32).tolist())
                y.extend((np.ones(int(b - a), dtype=np.int32) * i).tolist())

            for i in range(int(outer_upper_left[1][0]), int(inner_upper_left[1][0] + 1)):
                extension(outer_upper_left[0](i), outer_lip[0](i) + 1, i)
            for i in range(int(inner_upper_left[1][0]), int(outer_upper_left[1][-1] + 1)):
                extension(outer_upper_left[0](i),
                          inner_upper_left[0](i) + 1, i)
                extension(inner_lip[0](i), outer_lip[0](i) + 1, i)

            for i in range(int(inner_upper_right[1][-1]), int(outer_upper_right[1][-1] + 1)):
                extension(outer_upper_right[0](i), outer_lip[0](i) + 1, i)
            for i in range(int(inner_upper_right[1][0]), int(inner_upper_right[1][-1] + 1)):
                extension(outer_upper_right[0](i),
                          inner_upper_right[0](i) + 1, i)
                extension(inner_lip[0](i), outer_lip[0](i) + 1, i)

            # x = []  # will contain the x coordinates of points on lips
            # y = []
            # print(x, y)

            val = color.rgb2lab(
                (img[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
            L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
            L1, A1, B1 = color.rgb2lab(
                np.array((b1 / 255., g1 / 255., r1 / 255.)).reshape(1, 1, 3)).reshape(3, )
            L2, A2, B2 = L1 - L, A1 - A, B1 - B
            val[:, 0] += L2
            val[:, 1] += A2
            val[:, 2] += B2

            img[x, y] = color.lab2rgb(val.reshape(
                len(x), 1, 3)).reshape(len(x), 3) * 255
            cv2.imwrite(settings.MEDIA_ROOT +
                        '/logo/imgcpy3.png', img)
            # latest = tryons_image(
            #     sess_id=str(request.POST['sess_id'])+'_103', input_image='logo/imgcpy3.png')
            # latest.save()
            # img_new = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(settings.MEDIA_ROOT +
            #             '/logo/imgcpy4.png', img_new)

            # Apply tryons here

            img1 = tryons_image.objects.filter(id=imgObject.id).update(
                output_image='logo/imgcpy3.png')
            queryset = tryons_image.objects.filter(id=imgObject.id)
            serializer = TryOnsSerializers(queryset, many=True)
            # print(serializer.data)
            return Response(serializer.data)
        else:
            print("not valid")
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
