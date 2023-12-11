# !pip install stanza sacrebleu evaluate --quiet

## Check for cuda
# !nvidia-smi

## Imports
import random
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import stanza
import evaluate
from spacy import displacy
from nltk.tokenize import word_tokenize

# set options
pd.set_option('max_colwidth', None) # show full text
random.seed(42)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

path = ""

## Data
train = pd.read_csv(f"{path}/train.tsv", sep="\t", dtype={"input": str, "output": str})
test = pd.read_csv(f"{path}/test.tsv", sep="\t", dtype={"input": str, "output": str})
dev = pd.read_csv(f"{path}/dev.tsv", sep="\t", dtype={"input": str, "output": str})
total = pd.read_csv(f"{path}/total.tsv", sep="\t", dtype={"input": str, "output": str})
print("Train", train.shape)
print("Test", test.shape)
print("Dev", dev.shape)
print("Total", total.shape)

## Preprocess
train.input = train.input.str.replace("@CN@","Bob")
train.output = train.output.str.replace("@CN@","Bob")
train.output = train.output.str.replace("@SCN@", "John")

print(train.sample(10))

"""### Classifying questions using rules

There are 4 message types invovled here:

- AskWH messages: includes wh-words such as who, what, when, where
- AskYN messages: includes phrases such as ask if, ask whether or questions starting with are, is, can
- Req messages: includes phrases like tell to, ask to, remind to, etc.
- Statement messages: includes tell that, message that, remind that

### Lets check if by checking of these above words we can cover all the question types in the data or not
"""

ask_wh = ["who", "what", "when", "where"]
ask_yn = ["ask if", "ask whether"]
ask_yn_starts_with = ("is", "are", "can", "could", "will")
ask_req = ["tell.*to", "ask.*to", "remind.*to", "ask.*for"]
statements = ["tell.*that", "message.*that", "remind.*that", "know.*that"]

"""### Ask WH Questions"""

## Identify WH questions
train_ask_wh_questions = train[train.input.str.contains('|'.join(ask_wh), regex=True)]
print("Ask WH Questions", train_ask_wh_questions.shape[0])
print(train_ask_wh_questions)

"""### Ask YN Questions"""

## Identify YN questions
train_ask_yn_questions_1 = train[train.input.str.contains('|'.join(ask_yn), regex=True)]
train_ask_yn_questions_2 = train[train.input.str.startswith(ask_yn_starts_with)]
train_ask_yn_questions = pd.concat([train_ask_yn_questions_1, train_ask_yn_questions_2])
print("Ask YN Questions", train_ask_yn_questions.shape[0])
print(train_ask_yn_questions)

"""### Ask Req Questions"""

## Identify Req questions
train_ask_req_questions = train[train.input.str.contains('|'.join(ask_req), regex=True)]
print("Ask Req Questions", train_ask_req_questions.shape[0])
print(train_ask_req_questions)

"""### Statements"""

## Identify Statements
train_ask_stmt_questions = train[train.input.str.contains('|'.join(statements), regex=True)]
print("Ask Statement Questions", train_ask_stmt_questions.shape[0])
print(train_ask_stmt_questions)

"""### Check if we have any common question in these categories"""

print(train_ask_wh_questions.index.intersection(train_ask_yn_questions.index).shape)

print(train_ask_req_questions.index.intersection(train_ask_wh_questions.index).shape)

print(train_ask_req_questions.index.intersection(train_ask_yn_questions.index).shape)

print(train_ask_stmt_questions.index.intersection(train_ask_yn_questions.index).shape)

print(train_ask_stmt_questions.index.intersection(train_ask_req_questions.index).shape)

print(train_ask_stmt_questions.index.intersection(train_ask_wh_questions.index).shape)

"""As we can observe there are many common question in Ask WH and Req messages, the paper uses TFIDF and a custom list of stop words to reduce these number of common questions, but for now I will ignore them and perform processing for each one of them individually."""

train["ask_wh"] = train.input.str.contains('|'.join(ask_wh), regex=True)
train["ask_yn"] = train.input.str.contains('|'.join(ask_yn), regex=True) | train.input.str.startswith(ask_yn_starts_with)
train["ask_req"] = train.input.str.contains('|'.join(ask_req), regex=True)
train["stmt"] = train.input.str.contains('|'.join(statements), regex=True)
print(train.head())

"""### Total how many of these question from the total were we able to classify?"""

print(len(train_ask_wh_questions) + len(train_ask_yn_questions) + len(train_ask_req_questions) + len(train_ask_stmt_questions), len(train))

"""### POS Tagging

The POS tagging will help identify direct questions from indirect ones, the indriect question come with the label SQ tagging. The sentence level tags (S, S', SQ) are crucial in determining whether the word order between the subject and the auxillary is reversed. The word level tags (VB, VBP, VBZ0 indicate which verb needs to be changed as part of POV conversion.
"""

## NLTK POS Tagging
random_sentence = random.choice(train["input"].values)

words = nltk.word_tokenize(random_sentence)

tagging = nltk.pos_tag(words)
print(tagging)

## Standford nlp
stanford_nlp = stanza.Pipeline('en')
stanford_doc = stanford_nlp(random_sentence)

print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in stanford_doc.sentences for word in sent.words], sep='\n')

## Spacy
nlp = spacy.load("en_core_web_sm")
# doc = nlp(random_sentence)
doc = nlp("ask Bob, when are you coming for dinner")
for token in doc:
    print(token.text, token.pos_, token.tag_, token.dep_)
displacy.render(doc, style = "dep", jupyter=True)

"""### Constituent Parsing

| POS Tag | Info |

| S | Simple Declarative Clause |

| SBAR (S') | Clause introduced by subordinating conjunction |

| SQ | Inverted yes/no question |

| VB | verb, base form |

| VBP | verb, non-3rd person singular present |

| VBZ | verb, third person singular present |

| VP | verb phrase |
"""

## Constituency parsing
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
# random_sentence = random.choice(train["input"].values)
random_sentence = "ask Bob, when are you coming for dinner"
doc = nlp(random_sentence)
for sentence in doc.sentences:
    print(sentence.constituency)

## Constituency tree - Direct example
tree = doc.sentences[0].constituency
print(tree
      0

## Constituency tree - Indirect example
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
doc_2 = nlp("ask Bob, when he is coming for dinner")
tree_2 = doc_2.sentences[0].constituency
print("Tree:")
print(tree_2)

## Check if it's a direct or indirect question
print("SQ" in str(tree_2), "SQ" in str(tree))

## Run over all examples to classify direct and indirect questions
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', verbose=False, use_gpu=True)
def check_direct(text):
  doc = nlp(text)
  tree = doc.sentences[0].constituency
  return "SQ" in str(tree)

train["is_direct"] = train.input.apply(lambda x: check_direct(str(x)))
print(train.head())

input_values = train.input.values[:10]
train["is_direct"] = np.vectorize(lambda x: check_direct(str(x)))(input_values)
print(train.head())


train.to_csv(f"{path}/train_processed.csv", index=False)

train_processed = pd.read_csv(f"{path}/train_processed.csv")
print(train_processed.shape)
print(train.is_direct.value_counts())
print(train_processed.head())

"""### Transformations

1. `Changing word order`:

This step only applies to direct questions in AskYN and AskWH messages. During this process, multiple types of grammatical changes may apply, including do-deletion, and subject-auxiliary reversal (are you → you are).
"""

## Reverse the subject-auxillary order
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example text message
random_direct_yn_row = train_processed.iloc[24927]

print("Original Message:", random_direct_yn_row.input)
print("Expected response:", random_direct_yn_row.output)

def change_word_order(sentence):
  # Tokenize the text
  doc = nlp(sentence)

  # Process each sentence
  for sentence in doc.sents:
      subject = None
      verb = None
      verb_form = None

      # Identify the subject and verb
      for token in sentence:
          if ("subj" in token.dep_) or ("nsubj" in token.dep_):
              subject = token
          if ("aux" in token.dep_) and (token.text.lower() in ["is", "are"]):
              verb = token
              verb_form = token.text.lower()
          if not verb and ("ccomp" in token.dep_):
              verb = token
              verb_form = token.text.lower()

      # Check for subject-verb disagreement and reorder if needed
      if subject and verb and (verb_form in ["is", "are"]):
          # Swap the subject and verb
          new_sentence = f"hey Bob, {subject} {verb} {sentence[max(subject.i, verb.i) + 1:]}"
          transformed_sentence = new_sentence
      else:
          # No correction needed
          transformed_sentence = sentence.text
  return transformed_sentence

transformed_sentence = change_word_order(random_direct_yn_row.input)
print("Transformed sentence:", transformed_sentence)

direct_yn_ids = [24, 25, 32569, 32587, 18019, 24502]
for sent in train_processed.loc[direct_yn_ids].itertuples():
  print("Org:",sent.input)
  print("Exp:", sent.output)
  print("Transformed:", change_word_order(sent.input))
  print()

"""2. `Swapping pronouns/contact names`:

We use rules to convert a first person (I) to a third person (he/she) and a third person (he/she) to a second person (you). In cases where the contact name resides inside the message content, the rules would find it and switch it with a second person pronoun.
"""

import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example text message
random_wh_row = train_processed.iloc[24927]

print("Original Message:", random_wh_row.input)
print("Expected response:", random_wh_row.output)

def swap_pronouns(sentence):
  # Tokenize the text
  doc = nlp(sentence)

  # Define a set of pronouns to consider
  first_person_pronouns = {"I": "he", "me": "him", "my": "his"}
  third_person_pronouns = {"he": "you", "she": "you", "him": "you", "his": "your", "hers": "you"}

  # Apply rules to modify pronouns
  modified_text = []
  for token in doc:
      if token.text in first_person_pronouns:
          modified_text.append(first_person_pronouns[token.text])
      elif token.text in third_person_pronouns:
          modified_text.append(third_person_pronouns[token.text])
      else:
          modified_text.append(token.text)

  # Generate the modified text
  modified_text_message = " ".join(modified_text)
  return modified_text_message

transformed_sentence = swap_pronouns(change_word_order(random_wh_row.input))
print("Transformed sentence:", transformed_sentence)

"""3. `Fixing verb agreement`:

This step is to make sure the main verb/auxiliary agrees with the converted subject pronoun in person and number. In sentences with present tense, if we switch the subject she to you, we must change the main verb to its base form (is → are, wants → want, VBZ → VBP) as well.
"""

import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example text message
text_message = "can you ask if she wants icecream ?"

def fix_verb_agreement(sentence):
  # Tokenize the text
  doc = nlp(sentence)


  verb_agreement_rules = {
      "is": "are",
      "wants": "want",
      "has": "have"
  }

  if "you" in sentence:
      for word, change in verb_agreement_rules.items():
        sentence = sentence.replace(word.lower(), change)

print("Org:", text_message)
fix_verb_agreement(text_message)

"""4. `Adding prepending rules to reconstructed message content`:

We add the source contact name and appropriate reporting verbs to the beginning of each output, among other things. Each type of message has a different set of prepend rules and the VA can randomly choose prepends in the same set to sound more spontaneous. For example, an AskYN message with a direct question would need a prepend rule like@SCN@ asks if or @SCN@ is wondering whether. Similarly, a Req messages might use @SCN@ asks you to or @SCN@ would like to remind you to as prepends.
"""

import random

# Example text message
random_wh_row = train_processed.iloc[32569]

print("Original Message:", random_wh_row.input)
print("Expected response:", random_wh_row.output)

def extract_message(text, prepend_suggestion):
  message = random.choice(["Hey", "Hi"]) + " Bob, John " + prepend_suggestion
  # text = str(text.replace("@CN@", "John")).replace("to", "").replace("that","")
  doc = nlp(text)
  end_index = [e.end_char for e in doc.ents if e.label_ == "PERSON"]
  if len(end_index) > 0:
    message += text[end_index[0]:].replace(",","")
  return message

def reconstruct_message(sentence, message_types):
  # Define prepend rules for different message types
  prepend_suggestions = [""]
  if message_types.ask_yn and message_types.is_direct:
    prepend_suggestions = ["asks if", "is wondering whether", "wants to know if", "would like to know whether"]
  elif message_types.ask_req:
    prepend_suggestions = ["asks you to", "would like to remind you to", "requests that"]
  elif message_types.stmt:
    prepend_suggestions = ["shares that", "provides information that", "tells you that"]

  # Randomly select a prepend from the rules for the given message type
  selected_prepend = random.choice(prepend_suggestions)

  # Extract the message
  message = extract_message(sentence, selected_prepend)

  return message

modified_sentence = reconstruct_message(swap_pronouns(change_word_order(random_wh_row.input)), random_wh_row)
print("Transformed:", modified_sentence)

## Apply all the preprocessing sequentially

# Step 1:
train_processed["transformed"] = train_processed.apply(lambda row: change_word_order(row.input) if row['ask_yn'] and row['is_direct'] else row.input, axis=1)
train_processed["transformed"] = train_processed.input.apply(swap_pronouns)
train_processed["transformed"] = train_processed.input.apply(fix_verb_agreement)
train_processed["transformed"] = train_processed.apply(lambda row: reconstruct_message(row.input, row), axis=1)
train_processed.sample(10)

# Save the transformations
train_processed.to_csv(f"{path}/train_processed.csv", index=False)

print(pd.read_csv(f"{path}/train_processed.csv").sample(10))

"""### Check the results for rule-based approach"""

metric = evaluate.load("sacrebleu")

print(metric.compute(predictions=train_processed.transformed.values, references=train_processed.output.values))

import random
random_row = train_processed.loc[random.randint(0,len(train_processed))]
predictions = [random_row.transformed]
references = [[random_row.output]]
print("Input:", random_row.input)
print("Output:", random_row.output)
print("Rule based output:", random_row.transformed)
metric.compute(predictions=predictions, references=references)

train_processed["bleu_score"] = train_processed.apply(lambda row: metric.compute(predictions=[row["transformed"]], references=[[row["output"]]])["score"], axis=1)

train_processed.to_csv(f"{path}/train_processed.csv", index=False)

train_processed = pd.read_csv(f"{path}/train_processed.csv")
print(train_processed.head())

## Distribution of BLEU scores for rule-based approach
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(train_processed.bleu_score, kde=True)
plt.subplot(1,2,2)
sns.boxplot(train_processed, y="bleu_score");

## Sort by the BLEU score
train_processed_sorted = train_processed.sort_values("bleu_score", ascending=False)

"""### Check examples that went wrong"""

print(train_processed_sorted.tail(20))

