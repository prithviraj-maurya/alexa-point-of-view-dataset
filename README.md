[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/converting-the-point-of-view-of-messages/machine-translation-on-alexa-point-of-view)](https://paperswithcode.com/sota/machine-translation-on-alexa-point-of-view?p=converting-the-point-of-view-of-messages)

### Huggingface hub link to play with the model: [prithviraj-maurya/alexa_converting_pov](https://huggingface.co/prithviraj-maurya/alexa_converting_pov)

## Navigating Conversation with Voice Assistants

### Abstract

  This paper delves into the intricate realm of point-of-view (POV)
  conversion in voice messages within the landscape of virtual
  assistants (VAs). VAs, such as Amazon Alexa and Google Assistant, play
  multifaceted roles, from executing tasks to facilitating interpersonal
  communication. The paper addresses the challenge of adapting the
  perspective of messages for diverse contexts and recipients, exploring
  rule-based approaches, sequence-to-sequence models, and advanced
  Transformers. The rule-based approach demonstrates reasonable
  performance, while more sophisticated models exhibit higher accuracy.
  The paper concludes by envisioning future enhancements, including
  quotation detection and paraphrase generation, to further refine the
  role of VAs as conversational intermediaries.

Author: Prithviraj Anil Maurya

### INTRODUCTION

In the vast landscape of virtual assistants (VAs), such as Amazon Alexa
and Google Assistant, cloud-based software systems are meticulously
designed to decipher spoken utterances, comprehend user intents, and
seamlessly execute requested actions. Virtual assistants serve a myriad
of functions, ranging from playing music, setting timers, and providing
encyclopedic information to controlling smart home devices. Moreover,
they facilitate user-to-user communication, enabling voice calls and
text messaging.

This paper zooms in on a specific aspect of VA interactions -- the
conversion of the point of view in messages. When users communicate
messages to VAs, there often arises a need to adapt the perspective of
these messages to suit different contexts and recipients. This task
becomes particularly challenging when dealing with both direct and
indirect messages, necessitating intricate natural language processing
to handle co-reference relations effectively. Furthermore, when VAs
function as intermediaries, the point of view conversion becomes
essential for maintaining the natural flow of human-robot interactions.
This paper delves into these challenges and explores various approaches
to address them, ultimately aiming to enhance the role of VAs as
intermediaries in conversational exchanges.

### Research Question

How can the point-of-view (POV) in voice messages within the context of
virtual assistants be effectively converted to suit diverse contexts and
recipients, and what approaches yield optimal results in this
challenging natural language processing task?

### BACKGROUND

In the evolving landscape of virtual assistants (VAs), such as Amazon
Alexa and Google Assistant, sophisticated cloud-based systems are
designed to interpret spoken language, discern user intentions, and
execute a myriad of requested actions. These VAs play multifaceted
roles, from simple tasks like playing music and setting reminders to
more complex functions like controlling smart home devices and
facilitating user-to-user communication through voice calls and text
messaging.

One particular aspect of VA interactions that presents a unique set of
challenges is the conversion of the point of view (POV) in messages.
When users communicate with VAs, adapting the perspective of messages
becomes essential for diverse contexts and recipients. This complexity
is particularly evident when dealing with both direct and indirect
messages, requiring nuanced natural language processing (NLP) techniques
to handle co-reference relations effectively.

Moreover, as VAs increasingly function as intermediaries in human-robot
interactions, the accurate conversion of POV becomes crucial for
maintaining the natural flow of conversations. In this paper, we delve
into the intricacies of this problem, exploring various approaches, from
rule-based methods to advanced Transformer models, to enhance the
capability of VAs as intermediaries in conversational exchanges.

###  METHODS 

#### Data

The dataset utilized for this research underwent meticulous collection
and verification through a collaborative effort involving Amazon
Mechanical Turk workers (Callison-Burch and Dredze, 2010) and internal
associates. The dataset was anonymized by systematically replacing names
with special tokens, such as \"@CN@\" for contact names and \"@SCN@\"
for source contact names.

The dataset comprises a total of 46,562 samples, ensuring a balanced
representation of utterance categories. For experimentation and model
development, a partitioning strategy allocated 70% of the dataset for
training, 15% for validation, and an additional 15% for testing
purposes.

  | **Input**                                   | **Output**                                           |
| ------------------------------------------- | ----------------------------------------------------- |
| can you remind \@CN@ to meet the beautician  | hi \@CN@, \@SCN@ reminds you to meet the beautician   |
| tell \@CN@ to select a spot for the picnic   | hi \@CN@, \@SCN@ asks you to select a spot for the picnic |
| tell \@CN@ to answer the call                | hi \@CN@, \@SCN@ requested you to answer the call    |


  Sample rows from the dataset

#### Methodology

The BLEU (BiLingual Evaluation Understudy) metric was adopted for the
automatic evaluation of a machine-translated text. BLEU scores, ranging
from 0 to 1, measure the similarity of machine-translated text to a set
of high-quality reference translations. The BLEU metric provides a
robust evaluation of model performance.

1\. **Snip Direct Messages**:

The easiest strategy is just snipping the actual message from the text
and passing just the message. For example, a text tells Bob, I'm late
for dinner extracting the message I'm late for dinner and passing it on
to the recipient. However, as easy as it sounds won't work for many
sentences such as \"Ask Bob, should I bring him dinner?\" would be
extracted as should I bring him dinner which doesn't sound natural at
all, and is not something you will text to another person in normal
conversations.

2\. **Rule-Based Approach**:

Step 1: Classification Messages are classified into four types: AskWH,
AskYN, Request, and Statement. A dataset of questions was curated and
used to train a Naive Bayes Classifier and Gradient Boosted Tree
Classifier, achieving an F1 score of 0.93.

Step 2: Identify Direct vs Indirect Questions POS tagging and dependency
parsing were employed to identify direct questions from indirect ones.

Step 3: Transformation Changing word order for questions (AskWH and
AskYN). Swapping pronouns to convert first person to third person and
vice versa. Adjusting verbs to agree with the modified pronouns. Adding
appropriate prepend statements based on the message type.

3\. **Seq2Seq Model (Encoder-Decoder)**:

A sequence-to-sequence (seq2seq) model is a neural network architecture
designed for tasks involving sequential data, where an input sequence is
transformed into an output sequence. In the context of POV conversion,
this model is used to encode a source voice message and generate a
target voice message with the adjusted point of view.

*Encoder-Decoder Architecture*: The seq2seq model comprises two main
components: an encoder and a decoder. The encoder processes the input
sequence (source voice message) and represents it in a fixed-size
internal state. The decoder then takes this internal state and generates
the output sequence (target voice message).

*RNN (Recurrent Neural Network)*: Recurrent layers are employed in both
the encoder and decoder to handle sequential dependencies. However,
traditional RNNs face challenges in capturing long-range dependencies
due to vanishing or exploding gradient problems.

*Bahdanau Self-Attention*: To address the limitations of traditional
RNNs, Bahdanau attention mechanism is introduced. This attention
mechanism allows the model to focus on different parts of the input
sequence when generating each element of the output sequence. Bahdanau
attention enhances the model's ability to capture context across various
positions in the input sequence.

4\. **Transformers**

Transformers, introduced by Vaswani et al. in the paper \"Attention is
All You Need,\" represent a paradigm shift in sequence-to-sequence
modeling. This architecture relies on self-attention mechanisms,
dispensing with the need for recurrent connections. The HuggingFace
library's Transformers package provides easy access to a variety of
pre-trained transformer models.

*T5-base Model*: T5 (Text-to-Text Transfer Transformer) is a
transformer-based model that is pre-trained for a variety of natural
language processing tasks. The \"base\" variant refers to a model with
moderate size and capability. In the context of POV conversion, the T5
model is fine-tuned on the dataset, leveraging its powerful language
understanding and generation capabilities.

*Fine-tuning*: Fine-tuning involves adapting a pre-trained model to a
specific task or dataset. In this case, the T5-base model is fine-tuned
on the POV conversion dataset. Fine-tuning allows the model to
specialize its knowledge for the particular nuances and patterns present
in the dataset.

*HuggingFace Hub*: After fine-tuning, the resulting model is uploaded to
the HuggingFace Hub. The Hub serves as a repository for sharing and
accessing pre-trained models, enabling other researchers and developers
to utilize the fine-tuned POV conversion model for their applications.

These varied approaches encompass both traditional machine learning and
cutting-edge deep learning techniques, allowing us to comprehensively
assess the strengths and limitations of different models in addressing
the unique challenges posed by LLM-generated content detection in the
context of academic writing.

#### Analysis

**Rule-based Results** The rule-based approach achieved a BLEU score of
40.72, indicating a moderate level of performance in POV conversion.
BLEU scores range from 0 to 100, with higher scores indicating better
alignment between the generated and reference texts. The breakdown of
precision scores for different n-gram matches (unigrams, bigrams,
trigrams, and 4-grams) reveals varying levels of accuracy, with higher
precision for unigrams and lower precision for larger n-grams.

**Seq2Seq Model (Encoder-Decoder) Results**:

The seq2seq model, utilizing an RNN encoder-decoder architecture with
Bahdanau self-attention, produced the following examples of POV
conversion:

Original: \"Can you ask cn to read the catalog,\"

Translated: \"Hi \@CN@, \@SCN@ wanted you to read the catalog.\"

Original: \"Can you ask cn how to play a proper cover drive,\"

Translated: \"Hi \@CN@, \@SCN@ is eager to know how to play a proper
cover drive.\"

Original: \"Ask cn, what punishment would you give someone who is on
welfare,\"

Translated: \"Hi \@CN@, \@SCN@ is asking what punishment would you give
someone who is on welfare illegally?\"

**Transformers (T5-base Model) Results**:

The T5-base model, fine-tuned on the dataset, achieved an evaluation
BLEU score of 65.90. This score suggests a higher level of performance
compared to the rule-based approach. BLEU scores above 60 are generally
considered indicative of good performance in machine translation tasks.

The comparison table summarizes the BLEU scores for different
experiments using distinct models. The T5-base model outperformed both
the initial rule-based experiment and the BART model in terms of BLEU
scores, indicating its effectiveness in POV conversion for the given
dataset. The choice of model architecture plays a crucial role in
achieving higher accuracy and naturalness in generated sequences.

| **Model**            | **Score (BLEU)** | **Original paper** |
| -------------------- | ---------------- | ------------------ |
| Snipping messages    | 37.2             | -                  |
| Rule-based           | 40.72            | 46.6               |
| Encoder-Decoder      | 64.21            | 55.7               |
| T5                   | 66.425           | 63.8               |

*Table 1: Results from different experiments*


### CONCLUSIONS

In the exploration of point-of-view (POV) conversion in voice assistant
interactions, we employed a rule-based approach, a seq2seq model with an
encoder-decoder architecture, and a Transformer model (T5-base)
fine-tuned on the dataset. Each approach aimed to enhance the
naturalness and context preservation of voice messages as they are
relayed through virtual assistants. Here are the key takeaways:

1\. Rule-Based Approach:

\- Achieved a moderate BLEU score of 40.72.

\- Provided a foundational understanding of the challenges and
complexities in POV conversion.

\- Demonstrated the importance of considering different n-gram matches
for precision.

2\. Seq2Seq Model (Encoder-Decoder):

\- Presented a step towards more sophisticated models.

\- Generated examples showed promising results in transforming voice
messages while maintaining coherence.

3\. Transformers (T5-base Model):

\- Outperformed the rule-based approach and the BART model with a BLEU
score of 65.90.

\- Leveraged the power of pre-trained language models and fine-tuning
for improved POV conversion.

\- Demonstrated the effectiveness of Transformer architectures in
handling complex language tasks.

Insights and Implications:

\- Model Performance: The T5-base model showcased superior performance,
emphasizing the efficacy of leveraging Transformer architectures for POV
conversion tasks.

\- Context Preservation: All approaches grappled with challenges in
preserving context, especially in nuanced language and complex sentence
structures.

\- Future Directions: The study opens avenues for further research,
including exploring advanced Transformer architectures, integrating
contextual embeddings, and addressing challenges related to diverse
language patterns.

\- User Experience: Enhanced POV conversion contributes to a more
natural and user-friendly interaction with voice assistants, paving the
way for improved user experiences.

As voice assistants become integral to daily life, the ability to
convert and convey messages effectively becomes crucial. The presented
approaches provide a foundation for advancing the capabilities of
virtual assistants in understanding and responding to user messages with
improved context awareness and natural language processing. The success
of the Transformer model suggests that state-of-the-art language models
hold significant promise in addressing the intricacies of POV conversion
in voice-based interactions.

### References

[Lee, I. G., Zu, V., Buddi, S. S., Liang, D., Kulkarni, P., & FitzGerald, J. G. (2020). Converting the point of view of messages spoken to virtual assistants.](https://arxiv.org/abs/2010.02600).

[Qader, Wisam & M. Ameen, Musa & Ahmed, Bilal. (2019). An Overview of
Bag of Words;Importance, Implementation, Applications, and Challenges.
200-204. 10.1109/IEC47844.2019.8950616.
](https://www.researchgate.net/publication/338511771_An_Overview_of_Bag_of_WordsImportance_Implementation_Applications_and_Challenges).

[Reddy, A. J., Rocha, G., & Esteves, D. (2018). Defactonlp: Fact
verification using entity recognition, TFIDF vector comparison and
decomposable attention. arXiv preprint
arXiv:1809.00509](https://arxiv.org/abs/1809.00509).

[Vimal, Bhartendoo. (2020). Application of Logistic Regression in
Natural Language Processing. International Journal of Engineering
Research and. V9. 10.17577/IJERTV9IS060095.
](https://www.researchgate.net/publication/342075482_Application_of_Logistic_Regression_in_Natural_Language_Processing).

[Wang, S., & Jiang, J. (2015). Learning natural language inference with
LSTM. arXiv preprint
arXiv:1512.08849.](https://arxiv.org/abs/1512.08849).

[Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert:
Pre-training of deep bidirectional transformers for language
understanding. arXiv preprint
arXiv:1810.04805.](https://arxiv.org/abs/1810.04805).

[Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal,
P., & Amodei, D. (2020). Language models are few-shot learners. Advances
in neural information processing systems, 33,
1877-1901.](https://arxiv.org/abs/2005.14165).



## Citation 
```
@inproceedings{iglee2020,
  author={Isabelle G. Lee and Vera Zu and Sai Srujana Buddi and Dennis Liang and Purva Kulkarni and Jack Fitzgerald},
  title={{Converting the Point of View of Messages Spoken to Virtual Assistants}},
  year=2020,
  booktitle={Findings of EMNLP 2020},
  doi={to-be-added},
  url={to-be-added}
}
```

```
Lee, I.G., et al. "Converting the Point of View of Messages Spoken to Virtual Assistants"
```

## License
Under CDLA-Sharing 1.0 License
