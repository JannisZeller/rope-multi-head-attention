import torch
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import torchtext.vocab as ttv
import string
from nltk.corpus import stopwords


tokenizer = get_tokenizer("basic_english")
stopwords_english = stopwords.words("english")


def build_vocab(trn_rawpipe, vocab_size):
    def yield_tokens(text_datapipe):
        for x in text_datapipe:
            text = "".join([char for char in x[1] if char not in string.punctuation])
            text = tokenizer(text)
            text = [word for word in text if word not in stopwords_english]
            yield text

    vocab = ttv.build_vocab_from_iterator(
        yield_tokens(trn_rawpipe), specials=["<pad>", "<unk>"], max_tokens=vocab_size
    )
    vocab.set_default_index(vocab(["<unk>"])[0])

    return vocab


def preprocess_text(text, max_seqlen, vocab):
    text = "".join([char for char in text if char not in string.punctuation])
    text = tokenizer(text)
    text = [word for word in text if word not in stopwords_english]
    tokens = vocab(text)
    # using "if t!=1" would omit unknown tokens, but this might construct
    #  0-length training instances
    tokens = [t for t in tokens]
    return tokens[:max_seqlen]


def preprocess_batch(batch, vocab, max_seqlen, device):
    padding_value = vocab(["<pad>"])[0]

    ## Extracting labels and text from batch
    labels = torch.tensor([instance[0] for instance in batch])
    texts = [instance[1] for instance in batch]

    ## Preprocessing tokens and extracting lengths
    tokens = [preprocess_text(text, max_seqlen, vocab) for text in texts]
    tokens = pad_sequence(
        [torch.tensor(tokens_) for tokens_ in tokens],
        padding_value=padding_value,
        batch_first=True,
    )
    tokens = tokens.long().to(device)
    labels = labels.float().to(device) - 1.0

    ## Assembling output dict
    out_dict = {"labels": labels, "tokens": tokens}
    return out_dict
