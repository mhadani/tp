from dataclasses import dataclass

from typing import List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize

import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import LabelBinarizer


class FTVectorizer():
    """
    Classe pour vectoriser les textes et les labels
    - Les vecteurs de représentation des textes constitueront l'entrée du modèle neuronal
    - Les vecteurs de représentation des labels sont utilisées pour calculer la valeur du loss lors
    de l'entraînement du modèle
    Dans cet exemple, on définit cette classe FTVectorizer pour une représentation de type sac-de-mots
    (Bag-of-Words ou BoW). Pour utiliser une vectrisation de type modèle de langage pré-entraîné (par
    exemple avec le modèle transformers 'cammenbert') il faut modifier le code ci-dessous.

    """

    def __init__(self):
        self.vectorizer = CountVectorizer(
            analyzer="word",  # on fait sur la base des mots, possible de choisir caractère
            strip_accents="unicode",
            # dire si on veut normaliser les accents, les enlever ou pas, en français plus grande impact que englais
            tokenizer=self.tokenize,  # pour représenter une phrase ou un texte il faut tokeniser
            stop_words=None,  # or 'english' or a list of stopwords to ignore
            # ici on specifie none mais on aurait pu specifier la liste de stopwords crée plus tot
            # on pourrait inclure ici la ponctuation comme stopwords
            ngram_range=(1, 2),  # n-gram interval
            binary=True,  # binary values or frequencies?
            max_features=8000,  # max number of features (size of the vectorizer vocabulary)
            preprocessor=None
            # taille d'entrée du resau de neurone
        )
        self.label_binarizer = LabelBinarizer()


    def tokenize(self, text):
        """
        Customized tokenizer.
        Here you can remove, change or add other linguistic processing for generating the features
        """
        tokens = word_tokenize(text)
        tokens = [t.lower() for t in tokens]  # normalisation dela casse: mise ne minuscules (maj --> min)
        # On peut aussi tester le filtrage des symboles de ponctuation, lemmatisation, stemming etc
        # Par exemple, racinisation:
        # tokens = stemmer.stem(tokens)
        return tokens



    #############################################################################################
    # Ne pas modifier la signature (nom, arguments et type retourné) des méthodes suivantes
    # (mais vous pouvez modifier leur corps)
    #############################################################################################

    def fit(self, train_texts: List[str], train_labels: List[str]):
        self.vectorizer.fit(train_texts)
        self.label_binarizer.fit(train_labels)

    def input_size(self) -> int:
        """
        :return: The size of input vector representations
        """
        return len(self.vectorizer.vocabulary_)

    def output_size(self) -> int:
        """
        :return: The size of the output (label) vector representations
        """
        return len(self.label_binarizer.classes_)


    def vectorize_input(self, texts) -> List[torch.Tensor]:
        """
        Produces the vectorized representations of the input: these vectors will be the inputs to the
        neural network model
        :param texts:
        :return:
        """
        vects = self.vectorizer.transform(texts).toarray()
        return [torch.from_numpy(vect).float() for vect in vects]

    def vectorize_labels(self, labels) -> List[torch.Tensor]:
        """
        Produces the vectorized representations of the labels: these vectors will be the inputs to the
        the loss function used when training the neural network model
        :param labels:
        :return:
        """
        vects = self.label_binarizer.transform(labels)
        return [torch.from_numpy(vect).float() for vect in vects]

    def devectorize_labels(self, prediction_vects):
        return self.label_binarizer.inverse_transform(prediction_vects)

    def batch_collate_fn(self, batch_list):
        """

        :param batch_list:
        :return: the batch built from the list of examples batch_list
        """
        # batch_list is a list of tuples, each returned by the __get_item__() function of
        # the ReviewDataset class
        # create 2 separate lists for each element type in the tuples
        input_vects, label_vects = tuple(zip(*batch_list))
        # the batch will be a dictionary of tensors: a tensor for the input vectors, and another for the label_ids if any
        batch = dict({})
        batch['input_vects'] = torch.stack(input_vects)
        if label_vects[0] is not None:
            batch['label_vects'] = torch.stack(label_vects).float()
        # return the batch as a dictionary of tensors
        return batch



@dataclass
class HyperParameters:
    batch_size: int = 10
    learning_rate: float = 1e-3
    max_epochs: int = 20
    dropout: float = 0.5
    # early stopping
    es_monitor: str = 'val_loss'
    es_mode: str = 'min'
    es_patience: int = 5
    es_min_delta: float = 0.0
    # checkpoint save and selection
    ckpt_monitor: str = 'val_loss'
    ckpt_mode: str = 'min'



HP = HyperParameters()


class FTClassifier(pl.LightningModule):

    #############################################################################################
    # Ne pas modifier la signature (nom, arguments et type retourné) des méthodes suivantes
    # Pour améliorer le modèle, modifiez seulement le contenu des méthodes (vous pouvez
    # rajouter de nouvelles méthodes si nécessaire)
    #############################################################################################

    def __init__(self, vectorizer: FTVectorizer):
        super().__init__()
        self.vectorizer = vectorizer
        input_size = vectorizer.input_size()
        output_size = vectorizer.output_size()
        # hf_plm_name = HuggingFace Pretrained Language Model name
        # Linear layer(s) for the classifier component
        self.fcn = torch.nn.Sequential(
            # torch.nn.Linear(input_size, input_size*2),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(HP.dropout),
            torch.nn.Linear(input_size, output_size),
        )
        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, batch):
        out = self.fcn(batch['input_vects'])
        return out

    def training_step(self, batch, batch_idx):
        # training_step is called in PyTorch Lightning train loop
        y_hat = self.forward(batch)
        loss = self.loss_fn(y_hat, batch['label_vects'])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=HP.learning_rate)
        return optimizer


    def validation_step(self, batch, batch_ix):
        # validation_step is called in PyTorch Lightning train loop
        y_hat = self.forward(batch)
        loss = self.loss_fn(y_hat, batch['label_vects'])
        self.log_dict({'val_loss': loss.item()},
                      on_step=False, on_epoch=True, reduce_fx='mean', prog_bar=True,
                      )
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        y_hat = self.forward(batch)
        y_hat = F.softmax(y_hat, dim=1)
        return y_hat






