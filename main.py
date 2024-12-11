from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

app = Flask(__name__)

# Predefined context for Hugging Face QA
context = """
TCA provides consulting in digital strategies, marketing, and business growth.
It focuses on enhancing leadership in digital marketing, branding, and web development.
TCA differentiates itself by offering personalized services and fostering long-term relationships.
Our marketing strategy development services are designed to provide you with a clear roadmap for achieving your business objectives. We combine in-depth market research, competitor analysis, and industry insights to develop customized strategies that set you apart from the competition and maximize your marketing efforts.
When it comes to consultation services, choosing us means accessing a wealth of expertise and knowledge. Our team of highly skilled consultants brings a proven track record of delivering successful outcomes for clients.
This course will guide you on how to effectively build and promote your personal brand in the digital marketing landscape. Learn the strategies and tactics necessary to create a strong online presence and establish yourself as an expert in your field.
Let me know how I can assist you with our services or provide more information!
"""

# Load the Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


# Function to get the best matching response using TF-IDF
def find_best_match(user_input, paragraphs):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(paragraphs)
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X).flatten()
    best_match_idx = similarities.argmax()
    return paragraphs[best_match_idx]


# Function to get an answer from the Hugging Face QA pipeline
def get_huggingface_response(user_message, context):
    try:
        qa_result = qa_pipeline(question=user_message, context=context)
        return qa_result['answer']
    except Exception as e:
        print(f"Error using Hugging Face pipeline: {e}")
        return "I'm sorry, I couldn't process your request at the moment."


# Function to get a bot response
def get_bot_response(user_message):
    greetings = ['hi', 'hey', 'hello', 'good morning', 'good evening']
    user_message_lower = user_message.lower()

    # Check for greetings
    if any(greeting in user_message_lower for greeting in greetings):
        return "Hello! How can I help you today?"

    # Check for words present in the context and respond accordingly
    context_paragraphs = context.split('\n')  # Split context into paragraphs for better matching
    best_match = find_best_match(user_message, context_paragraphs)

    if best_match:
        return best_match
    
    # Use Hugging Face QA for a detailed answer
    return get_huggingface_response(user_message, context)


# Flask Routes
@app.route('/')
def home():
    default_questions = [
        "What services does TCA provide?",
        "How can I contact TCA?",
        "What makes TCA different from others?",
        "Tell me more about TCA's digital marketing strategies."
    ]
    return render_template('index.html', questions=default_questions)



@app.route('/chat', methods=['POST'])
def chat_message():
    try:
        data = request.get_json()
        user_message = data.get('user_message')

        if not user_message:
            return jsonify({'reply': "I'm sorry, I didn't understand that."}), 400

        # Get the bot response
        response = get_bot_response(user_message)
        return jsonify({'reply': response})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({'reply': "There was an error processing your request."}), 500


if __name__ == '__main__':
    app.run(debug=True)
