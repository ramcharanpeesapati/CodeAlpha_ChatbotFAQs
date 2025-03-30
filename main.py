import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('faq_data.json', 'r') as file:
    faq_data = json.load(file)

nlp = spacy.load('en_core_web_sm')

questions = [faq['question'] for faq in faq_data['faqs']]
answers = [faq['answer'] for faq in faq_data['faqs']]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


def get_response(user_input):
    user_input_doc = nlp(user_input)
    processed_input = ' '.join([token.lemma_ for token in user_input_doc])

    user_input_vector = vectorizer.transform([processed_input])

    similarities = cosine_similarity(user_input_vector, question_vectors)

    most_similar_index = similarities.argmax()

    return answers[most_similar_index]


if _name_ == "_main_":
    print("FAQ Chatbot: How can I help you today? (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_response(user_input)
        print(f"Bot: {response}")
