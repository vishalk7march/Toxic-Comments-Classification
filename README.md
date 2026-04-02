# Toxic Comments Classification Using a Pre-Trained Model

This project classifies a sentence into the 6 Jigsaw toxic comment categories:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

It uses a pre-trained model through `Detoxify`, which is built for the Toxic Comment Classification Challenge label set. If the input is toxic, the app also rewrites the sentence into a safer, less harmful version. If the input is already normal, it leaves the sentence unchanged.

## Dataset

Hugging Face dataset:

- `thesofakillers/jigsaw-toxic-comment-classification-challenge`

Note:
If you typed `hesofakillers`, the correct dataset owner is `thesofakillers`.

## Install

```bash
pip install -r requirements.txt
```

## Run The Classifier

Interactive mode:

```bash
python app.py
```

Single sentence:

```bash
python app.py --text "You are an idiot and I hate you"
```

JSON output:

```bash
python app.py --text "You are an idiot and I hate you" --json
```

Custom threshold:

```bash
python app.py --text "You are an idiot" --threshold 0.35
```

## Run The GUI

```bash
python gui.py
```

The GUI lets you:

- type a sentence in a text box
- click `Analyze Sentence`
- see the toxic rating
- see all 6 category scores
- view the safer rewritten sentence

## View Dataset Information

```bash
python dataset_info.py
```

## Example Output

```text
Input sentence      : You are an idiot and I hate you
Toxic rating        : 96.41/100
Detected categories : insult, toxic
Category scores:
  - toxic         0.9641
  - severe_toxic  0.2114
  - obscene       0.1042
  - threat        0.0121
  - insult        0.8812
  - identity_hate 0.0059
Safe sentence       : you seem an person and I do not like you
```

## Files

- `app.py`: main toxic comment classifier
- `dataset_info.py`: prints dataset split and column details
- `requirements.txt`: Python dependencies
