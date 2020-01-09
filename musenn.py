#!/usr/bin/python3
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(prog="musenn.py", description="Minimal Character Recurent Neural Network for Music")
parser.add_argument('--hidden-size', '-S', help='Size of the hidden layer (hyperparameter, default: 100).', default=100, type=float)
parser.add_argument('--seq-length', '-L', help='Numbers of steps to unroll the RNN for (hyperparameter, default: 25).', default=25, type=float)
parser.add_argument('--learning-rate', '-R', help='How much the network should learn (hyperparameter, default: .1).', default=.1, type=float)
parser.add_argument('--threshold', '-t', help='The minimal loss to reach before stopping (default: .01)', default=.01, type=float)
parser.add_argument('--max-iter', '-m', help='The maximum number of iterations (infinite: -1, default: -1)', default=-1, type=float)
parser.add_argument('--chkpt', '-c', metavar='n', help='Saves sample every n iteration (default: 20000, never: -1).', default=20000, type=float)
parser.add_argument('--show-every', '-p', metavar='n', help='Prints sample to console every n iteration (default:2000, never: -1).', default=2000, type=float)
parser.add_argument('--length', '-l', help='The length of the sample at checkpoint (default: 500).', default=500, type=float)
parser.add_argument('--file', '-f', help='Input file to train on (default: "input.txt").', default="input.txt", type=str)
parser.add_argument('--foldern', '-o', metavar='folder_name', help='Output directory will be "output-folder_name" (default: random).', default=str(np.random.randint(8999)+1000), type=str)
parser.add_argument("--key", "-k", help='Use this to force the output notes to be in a certain key (default: all)', default="all", type=str)
args = parser.parse_args()

# data I/O
data = open(args.file, 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = int(float(args.hidden_size)) # size of hidden layer of neurons
seq_length = int(float(args.seq_length)) # number of steps to unroll the RNN for
learning_rate = float(args.learning_rate)
threshold = float(args.threshold) # the loss goal
max_iter = int(float(args.max_iter))
chkpt = int(float(args.chkpt))
showevery = int(float(args.show_every))
length = int(float(args.length))
foldern = args.foldern

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(list(range(vocab_size)), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

if not os.path.exists('output-' + foldern):
    os.makedirs('output-' + foldern)

if not os.path.exists('chkpts'):
    os.makedirs('chkpts')

while smooth_loss > threshold and (n <= max_iter or max_iter == -1):
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % showevery == 0 and showevery != -1:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  if n % chkpt == chkpt - 1:
    sample_ix = sample(hprev, inputs[0], length)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    with open('output-' + foldern + '/epoch ' + str(int((n+1)/chkpt)) + ', loss ' + str(smooth_loss) + '.txt', 'w+') as f:
        if args.key != "all":
            txt = txt.replace("^","").replace("_","")
            f.write("K:"+args.key+"\n")
        f.write(txt)
    f.close()
    print('[Sample saved to : output-' + foldern + '/epoch ' + str(int((n+1)/chkpt)) + ', loss ' + str(smooth_loss) + '.txt]')
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 

sample_ix = sample(hprev, inputs[0], length)
txt = ''.join(ix_to_char[ix] for ix in sample_ix)
with open('output-' + foldern + '/[FINAL] epoch ' + str(int((n+1)/chkpt)) + ', loss ' + str(smooth_loss) + '.txt', 'w+') as f:
    f.write(txt)
f.close()
print('[FINAL sample saved to : output-' + foldern + '/[FINAL] epoch ' + str(int((n+1)/chkpt)) + ', loss ' + str(smooth_loss) + '.txt]')
