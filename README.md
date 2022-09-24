# :vs: Adversarial Natural Language Processing :newspaper: :pencil:
*Deep Neural Networks* can be compromised by attacks that lead them to wrong classification of instances which have been modified through small perturbations. These instances are called *Adversarial Examples*. In the *Natural Language Processing (NLP)* field, unlike other applications, a minimal perturbation of an instance can be evident to the human eye: editing a singular word in the corpus of a text can lead to variation or loss of its original meaning.

This thesis project follow the experiment of the paper [*Generating Natural Language Adversarial Examples*](https://aclanthology.org/D18-1316/) and proposes the use of a genetic optimization algorithm that generates Adversarial Examples which maintain a syntactic and semantic meaning similar to the one of the original text, while managing to trick a DNN *Sentiment Analysis* model.

## Repository Structure
    .
    ├── src
    │   ├── ...
    │   ├── utils
    │   │   ├── attacker.py                     # Script containing the class of the algorithm performing the adversarial attack
    │   │   ├── black_box.py                    # Script containing the class of the Black Box Sentiment Analysis model
    │   │   ├── black_box_preprocessing.py      # Script containing the class of the preprocesser for the Black Box model
    │   │   └── seed_setter.py                  # Script to set the seed for riproducibility
    │   ├── 00 IMDB Dataset Download and Analisys.ipynb
    │   ├── 01 IMDB Data Preprocessing for the GloVe Embeddings Vectors.ipynb
    │   ├── 02 Creating a DNN for IMDB Sentiment Analysis and the Black-Box Algorithm.ipynb 
    │   ├── 03 Words Distance Matrix and Google One Billion Words Language Model.ipynb
    │   ├── 04 Adversarial Attack.ipynb
    │   └── 05 Test Results.ipynb
    ├── tesi.pdf                                # The thesis on the project (in Italian)
    ├── .gitignore
    ├── LICENSE
    └── README.md

## Versioning

Git is used for versioning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 

