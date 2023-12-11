[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/converting-the-point-of-view-of-messages/machine-translation-on-alexa-point-of-view)](https://paperswithcode.com/sota/machine-translation-on-alexa-point-of-view?p=converting-the-point-of-view-of-messages)

# Point of View (POV) of Message Conversion Dataset
Virtual assistants (VAs) tend to be literal in their delivery of messages. Most likely, if you ask them to deliver a message, the VAs either send a recorded message or a literal transcription to the receiver. To make incremental improvement towards a virtual assistant that you may speak to conversationally and naturally, we have provided the data necessary to build AI systems that can convert the point of view of messages spoken to virtual assistants.

If a sender asks the virtual assistant to relay a message to a receiver, the virtual assistant converts the message to VA's perspective and composes a conversational relay message. i.e.

- **Sender (isabelle):** ask nick if he wants anything from trader joe's
- **VA to receiver (nick):** hi nick, isabelle wants to know if you want anything from trader joe's? 

Our public release of the dataset contains parallel corpus of input and output utterances as such. This release also contains surveys used for data collection and human evaluation on resulting model output covered in our paper.

### Table of Content
1. [Data](#data)
4. [Citation](#citation)
5. [Acknowlegements](#acknowledgements)


## Data 

The dataset contains parallel corpus of input (`input` column) message and POV converted messages (`output` column). An example of a pair is
```tell @CN@ that i'll be late [\t] hi @CN@, @SCN@ would like you to know that they'll be late.``` The input and pov-converted output pair is tab separated. `@CN@` tag is a placeholder for the contact name (receiver) and `@SCN@` tag is a placeholder for source contact name (sender).

The total dataset has 46563 pairs. This data is then test/train/dev split into 6985 pairs/32594 pairs/6985 pairs.


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

## Acknowledgements 
We'd like to thank Steven Spielberg P. for coordinating our efforts and for early contribution, and we'd like to thank Adrien Carre and Minh Nguyen on coordinating with associates for the dataset and human evaluation of model output.


## License
Under CDLA-Sharing 1.0 License
