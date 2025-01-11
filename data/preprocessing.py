import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imports import WordNetLemmatizer, stopwords, BertTokenizer, BertModel, nltk, np, torch, train_test_split, json, pickle

# Initialize the lemmatizer, BERT tokenizer, and BERT model
lemmatizer = WordNetLemmatizer()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Download NLTK packages
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Initialize lists to store words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',', '😊', '👋', '🚀']

# Load intents JSON file
with open('data/database.json', 'r') as file:
    intents = json.load(file)
    
# Process each pattern in intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:

        stop_words = set(stopwords.words('spanish'))
        word_list = nltk.word_tokenize(pattern)
        word_list = [word for word in word_list if word.lower() not in stop_words]
        word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in ignore_letters]

        words.extend(word_list)
        documents.append((pattern, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Save words and classes to pickle files
with open('data/words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('data/classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Create training data
X_train = []
y_train = []
output_empty = [0] * len(classes)

for document in documents:
    lemmatized_sentence = " ".join(word_list)
    inputs = tokenizer(document[0], return_tensors='pt', padding=True, truncation=True, max_length=50)

    with torch.no_grad():
        embeddings = bert_model(**inputs).last_hidden_state.mean(dim=1).numpy()

    X_train.append(embeddings.flatten())
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    y_train.append(output_row)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Save training and validation data to .npy files
np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)
np.save('data/X_val.npy', X_val)
np.save('data/y_val.npy', y_val)