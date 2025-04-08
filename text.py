1-parse tree

# Define the grammar rules properly
grammar = {
    'S': [['NP', 'VP']],
    'NP': [['Det', 'Nom']],
    'VP': [['V', 'NP']],
    'Nom': [['Adj', 'Nom'], ['N']],
    'Det': [['the']],
    'Adj': [['little'], ['angry'], ['frightened']],
    'N': [['squirrel'], ['bear']],
    'V': [['chased']]
}

# Tokenized input sentence
sentence = "the angry bear chased the frightened little squirrel".split()
index = 0  # Global index to track the current word


# Recursive parse function
def parse(symbol):
    global index
    # Terminal
    if symbol not in grammar:
        if index < len(sentence) and sentence[index] == symbol:
            node = (symbol,)
            index += 1
            return node
        else:
            return None
    # Non-terminal
    for rule in grammar[symbol]:
        saved_index = index
        children = []
        for part in rule:
            result = parse(part)
            if result is None:
                index = saved_index
                break
            children.append(result)
        else:
            return (symbol, children)
    return None


# Helper to print parse tree
def print_tree(node, indent=0):
    if isinstance(node, tuple):
        if len(node) == 1:
            print('  ' * indent + node[0])
        else:
            print('  ' * indent + node[0])
            for child in node[1]:
                print_tree(child, indent + 1)


# Start parsing from 'S'
tree = parse('S')

if tree and index == len(sentence):
    print("✅ Parse successful! Here's the parse tree:\n")
    print_tree(tree)
else:
    print("❌ Failed to parse the sentence.")

2-parsetree

# Grammar based on the image
grammar = {
    'S': [['NP', 'VP']],
    'NP': [['Det', 'Nominal'], ['N']],
    'VP': [['V', 'NP']],
    'Det': [['A']],
    'Nominal': [['N']],
    'N': [['Restaurant'], ['Dosa']],
    'V': [['Serves']]
}

# Tokenized input
sentence = "A Restaurant Serves Dosa".split()
index = 0


# Parse function
def parse(symbol):
    global index
    # Terminal
    if symbol not in grammar:
        if index < len(sentence) and sentence[index] == symbol:
            node = (symbol,)
            index += 1
            return node
        else:
            return None
    # Non-terminal
    for rule in grammar[symbol]:
        saved_index = index
        children = []
        for part in rule:
            result = parse(part)
            if result is None:
                index = saved_index
                break
            children.append(result)
        else:
            return (symbol, children)
    return None

tree = parse('S')

if tree and index == len(sentence):
    print("✅ Parse successful! Here's the parse tree:\n")
    print_tree(tree)
else:
    print("❌ Failed to parse the sentence.")

3-Minimum edit distance
# Get input from user
string1 = input("Enter first string: ")
string2 = input("Enter second string: ")

m = len(string1)
n = len(string2)

# Create a matrix for storing results
dp = [[0 for j in range(n+1)] for i in range(m+1)]

# Fill the matrix
for i in range(m+1):
    for j in range(n+1):
        if i == 0:
            dp[i][j] = j  # Insert all characters
        elif j == 0:
            dp[i][j] = i  # Remove all characters
        elif string1[i-1] == string2[j-1]:
            dp[i][j] = dp[i-1][j-1]
        else:
            insert = dp[i][j-1]
            delete = dp[i-1][j]
            replace = dp[i-1][j-1]
            dp[i][j] = 1 + min(insert, delete, replace)

# Print final result
print("Minimum Edit Distance:", dp[m][n])

4 - Trigram most probable word
# Corpus
corpus = [
    "<s> I am Henry </s>",
    "<s> I like college </s>",
    "<s> Do Henry like college </s>",
    "<s> Henry I am </s>",
    "<s> Do I like Henry </s>",
    "<s> Do I like college </s>",
    "<s> I do like Henry </s>"
]

# Function to build trigram model
def build_trigram_model(corpus):
    trigram_counts = {}

    for sentence in corpus:
        tokens = sentence.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            key = (w1, w2)
            if key not in trigram_counts:
                trigram_counts[key] = {}
            if w3 not in trigram_counts[key]:
                trigram_counts[key][w3] = 0
            trigram_counts[key][w3] += 1

    return trigram_counts

# Build trigram frequency
trigram_model = build_trigram_model(corpus)

# Given context
context = ("Do", "I", "like")
bigram = (context[1], context[2])

# Predict next word
if bigram in trigram_model:
    next_word = max(trigram_model[bigram], key=trigram_model[bigram].get)
    print(f"Most probable next word (without inbuilt): {next_word}")
else:
    print("No match found.")

5 - Stemming lemmatization
# Simple Rule-Based Stemmer
def stem(word):
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[: -len(suffix)]
    return word

# Simple Dictionary-Based Lemmatizer
def lemmatize(word):
    lemma_dict = {
        'programming': 'program',
        'loving': 'love',
        'lovely': 'love',
        'kind': 'kind'  # already base form
    }
    return lemma_dict.get(word.lower(), word)

# Test Words
words = ['Programming', 'Loving', 'Lovely', 'Kind']

# Output
print("Word\t\tStem\t\tLemma")
print("-" * 40)
for w in words:
    original = w
    stemmed = stem(w.lower())
    lemmatized = lemmatize(w)
    print(f"{original:<12}{stemmed:<12}{lemmatized}")

6 - pos tagging
# 1. Lexicon (expandable)
lexicon = {
    "i": "PRON", "we": "PRON", "you": "PRON", "he": "PRON", "she": "PRON", "they": "PRON", "us": "PRON", "your": "PRON",
    "need": "VERB", "permit": "VERB", "like": "VERB", "address": "VERB", "am": "AUX", "would": "MODAL", "is": "AUX",
    "to": "PART", "from": "PREP", "on": "PREP", "in": "PREP", "of": "PREP", "this": "DET", "a": "DET", "the": "DET",
    "flight": "NOUN", "public": "NOUN", "issue": "NOUN", "everything": "PRON", "shipping": "ADJ", "atlanta": "PROPN"
}

# 2. Suffix-based rules
def suffix_rules(word):
    if word.endswith("ing"):
        return "VERB"
    elif word.endswith("ed"):
        return "VERB"
    elif word.endswith("ly"):
        return "ADV"
    elif word.endswith("ous") or word.endswith("ful"):
        return "ADJ"
    elif word.endswith("ion") or word.endswith("ment"):
        return "NOUN"
    return None

# 3. Capitalization-based heuristic
def capital_rule(word, position):
    if word[0].isupper() and position != 0:
        return "PROPN"  # Proper noun (not sentence start)
    return None

# 4. Master tagging function
def tag_word(word, position):
    word_lower = word.lower()
    # 1. Try lexicon
    if word_lower in lexicon:
        return lexicon[word_lower]
    # 2. Try suffix rule
    suffix_guess = suffix_rules(word_lower)
    if suffix_guess:
        return suffix_guess
    # 3. Capitalization
    cap_guess = capital_rule(word, position)
    if cap_guess:
        return cap_guess
    # 4. Fallback
    return "NOUN"

# 5. POS tagger function
def pos_tag(sentence):
    tokens = sentence.replace('.', '').split()
    tags = []
    for i, word in enumerate(tokens):
        tag = tag_word(word, i)
        tags.append((word, tag))
    return tags

# 6. Test sentences
sentences = [
    "I need a flight from Atlanta.",
    "Everything to permit us.",
    "I would like to address the public on this issue.",
    "We need your shipping address."
]

# 7. Display results
for i, sent in enumerate(sentences, 1):
    print(f"\nSentence {i}: {sent}")
    for word, tag in pos_tag(sent):
        print(f"{word:15} → {tag}")

7 - Text summarization
text = '''Artificial Intelligence is changing the world. 
It is used in healthcare to detect diseases. 
AI is used in education to personalize learning. 
It helps in self-driving cars. 
Many industries use AI for automation. 
In banking, AI detects frauds. 
AI chatbots improve customer service. 
Entertainment apps use AI for recommendations. 
AI is useful in agriculture and weather prediction. 
Still, AI raises ethical concerns.'''

sentences = text.split('. ')
freq = {}
for s in sentences:
    for w in s.lower().replace('.', '').replace(',', '').split():
        freq[w] = freq.get(w, 0) + 1
scores = []
for s in sentences:
    score = sum(freq.get(w, 0) for w in s.lower().split())
    scores.append((score, s))
top = sorted(scores, reverse=True)[:3]
for _, s in top:
    print(s.strip() + '.')

8 - regex
import re

# Sample text with 3 emails and 2 phone numbers
text = '''Contact us at sunil123@gmail.com or support@domain.org for queries. 
You can also reach ankit.k@xyz.in. Call 9876543210 or 9123456789 for urgent help.'''

# 1. Extract all email IDs
emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
print("Email IDs:", emails)

# 2. Extract all mobile numbers (Assuming 10-digit Indian numbers)
mobiles = re.findall(r'\b[6-9]\d{9}\b', text)
print("Mobile Numbers:", mobiles)

# 3. Extract names matching pattern: 'S u _ _ _'
names = ['Sunil', 'Shyam', 'Ankit', 'Surjeet', 'Sumit', 'Subhi', 'Surbhi', 'Siddharth', 'Sujan']
pattern_names = [name for name in names if re.fullmatch(r'Su\w{3}', name)]
print("Names matching 'Su___':", pattern_names)

# 4. Use of re.search()
result = re.search(r'\d{10}', text)
print("First phone found (search):", result.group() if result else "Not found")

# 5. Use of re.match()
m = re.match(r'Contact', text)
print("Match at start (match):", m.group() if m else "Not matched")

# 6. Use of re.sub()
clean_text = re.sub(r'\d{10}', 'XXXXXXXXXX', text)
print("Text after replacing numbers:", clean_text)

# 7. Use of re.compile()
pattern = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
compiled_emails = pattern.findall(text)
print("Emails using re.compile():", compiled_emails)

# 8. Match 'ab' followed by zero or more 'c'
test1 = ['ab', 'abc', 'abcc', 'ac', 'a', 'abb']
for word in test1:
    if re.fullmatch(r'abc*', word):
        print("ab followed by 0 or more c:", word)

# 9. Match 'a' followed by zero or more 'bc'
test2 = ['a', 'abc', 'abcbc', 'abcbcbc', 'ab', 'abbc']
for word in test2:
    if re.fullmatch(r'(abc)*a|a', word):
        print("a followed by 0 or more bc:", word)

# 10. Match 'ab' followed by zero or one 'c'
test3 = ['ab', 'abc', 'abcc', 'ac', 'abb']
for word in test3:
    if re.fullmatch(r'abc?', word):
        print("ab followed by 0 or 1 c:", word)
