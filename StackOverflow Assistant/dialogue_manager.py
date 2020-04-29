import os
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        n = int(len(thread_ids) / 2)


        # HINT: you have already implemented a similar routine in the 3rd assignment.

        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim).reshape(1,-1)
        b1, d1 = pairwise_distances_argmin_min(question_vec, thread_embeddings[:n, :])
        b2, d2 = pairwise_distances_argmin_min(question_vec, thread_embeddings[n:, :])

        if d1[0] <= d2[0]:
            best_thread = b1[0]
        else:
            best_thread = b2[0] + n
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")
        self.paths = paths
        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

        #init chatbot
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param

        self.chitchat_bot = ChatBot('Mushfiqul')
        self.trainer = ChatterBotCorpusTrainer(self.chitchat_bot)
        self.trainer.train("chatterbot.corpus.english")

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        # Intent recognition:
        self.intent_recognizer = unpickle_file(self.paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(self.paths['TFIDF_VECTORIZER'])
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)
        self.intent_recognizer = ''
        self.tfidf_vectorizer = ''

        # Chit-chat part:
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            response = self.chitchat_bot.get_response(question)
            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]

            try:
                # Pass prepared_question to thread_ranker to get predictions.
                thread_id = self.thread_ranker.get_best_thread(question, tag)

                return self.ANSWER_TEMPLATE % (tag, thread_id)
            except:
                return 'Memory size is to sort to load the details'
