import torch
import torch.optim as optim
import argparse
import math
import time

from model import *
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchnlp.metrics.bleu import get_moses_multi_bleu

import spacy

# Parse hyperparameters from args
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=.001)

args = parser.parse_args()

# Hyperparameters
BATCH_SIZE = args.batch_size
EMB_DIM = args.emb_dim
HID_DIM = args.hid_dim
N_LAYERS = args.n_layers
DROPOUT = args.dropout
N_EPOCHS = args.epochs

# Load spaCy French and English for tokenizer
spacy_fr = spacy.load('fr')
spacy_en = spacy.load('en')


# Tokenize French sentence
def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]


# Tokenize English sentence
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


# Split train / valid / test dataset
SRC = Field(tokenize=tokenize_fr, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
train_data, valid_data, test_data = Multi30k.splits(exts=('.fr', '.en'), fields=(SRC, TRG))

print("Number of training examples:", len(train_data.examples))
print("Number of validation examples:", len(valid_data.examples))
print("Number of testing examples:", len(test_data.examples))

print("First train data:", vars(train_data.examples[0]))
print("First valid data:", vars(valid_data.examples[0]))
print("First test data:", vars(test_data.examples[0]))

# Build the source and target dictionary with minimum frequency 2
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print("Unique tokens in source (fr) vocabulary:", len(SRC.vocab))
print("Unique tokens in target (en) vocabulary:", len(TRG.vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

# Build Model
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)
print(model)


# Initialize the weights of model
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)


# Count trainable parameters of model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("The model has ", count_parameters(model), " trainable parameters")


# Add padding to the target sentence
PAD_IDX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        h, r = [], []
        for batch in iterator:
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)

            bleu_trg = trg[1:].permute(1, 0)
            for t in bleu_trg:
                t = t.tolist()
                h.append(' '.join(map(str, t)))

            bleu_output = output[1:].permute(1, 0, 2)
            for o in bleu_output:
                _, indices = torch.max(o, 1)
                indices = indices.tolist()
                r.append(' '.join(map(str, indices)))

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

        print("BLEU:", get_moses_multi_bleu(h, r))
    return epoch_loss / len(iterator)


# Measure the elapsed time
def epoch_time(start, end):
    elapsed = end - start
    elapsed_mins = int(elapsed / 60)
    elapsed_secs = int(elapsed - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')

        print("Epoch: %2d | Time: %d m %d s" % (epoch, epoch_mins, epoch_secs))
        print("\tTrain Loss: %.3f | Train PPL: %7.3f" % (train_loss, math.exp(train_loss)))
        print("\tValid Loss: %.3f | Valid PPL: %7.3f" % (valid_loss, math.exp(valid_loss)))


    model.load_state_dict(torch.load('model.pt'))

    test_loss = evaluate(model, test_iterator, criterion)

    print("| Test Loss: %.3f | Test PPL: %7.3f |" % (test_loss, math.exp(test_loss)))