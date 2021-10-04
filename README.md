# SemEval2021
Final project for my *Natural Language Processing* class. Made in collaboration with [Mgczacki](https://github.com/Mgczacki) and [roher1727](https://github.com/roher1727).

## Toxic Spans Detection

The original dataset is in [Codalab](https://competitions.codalab.org/competitions/25623), it consists of around 10 thousand publications that come from [Civil Comments dataset](https://www.tensorflow.org/datasets/catalog/civil_comments), in which specific sections have been labeled as toxic language, the task is to be able to train a model that detects these toxic sections.

First, create a virtual environment:

```
virtualenv toxic-spans-venv
```

To activate it:

```
source toxic-spans-venv/bin/activate
```

Then clone this repository and install the requirements:

```
git clone https://github.com/davidguzmanr/SemEval2021.git
cd SemEval2021
pip install -r requirements.txt
```
