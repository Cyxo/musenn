# MuseNN - When RNN meets music

### Summary :

1.  [Some compositions](#comp)
2.  [How does it work](#func)
3.  [Contact MuseNN](#contact)

## Some composition

MuseNN was trained on different styles. The best results came from House melodies on piano and a Classical quatuor from Mozart.

The other styles I trained the RNN on resulted on complete mess of notes (almost like [Black MIDIs](https://www.youtube.com/watch?v=I906a5msynw)), or were too close from the training models. So here are the best I could get (training on little data - around 250 chars -, with a minimal RNN, on my highschool's server) :

<table>

<tbody>

<tr>

<td>► NNHouse</td>

<td>1:42</td>

<td>1 instrument (piano)</td>

<td>Trained on ten 15-seconds melodies</td>

</tr>

<tr>

<td>► Moznn</td>

<td>4:43</td>

<td>4 instruments (violin 1 and 2, viola, cello)</td>

<td>Trained on Mozart's [String quartet n°1 in G](https://www.youtube.com/watch?v=2I9LyiGrclc)</td>

</tr>

</tbody>

</table>

### Downloads :

MP3s :

*   [NNHouse (1.95MB)](http://www.mediafire.com/file/nkc82sz6y70mclt/NNHouse+-+Final.mp3)
*   [Moznn (5.41MB)](http://www.mediafire.com/file/c0bfuctnwrn5psc/MOZNN.mp3)

Scores :

*   [NNHouse (1.95MB)](http://www.mediafire.com/file/nkc82sz6y70mclt/NNHouse+-+Final.mp3)
*   [Moznn (5.41MB)](http://www.mediafire.com/file/c0bfuctnwrn5psc/MOZNN.mp3)

## How does it work ?

First things first : <big>What is MuseNN ?</big> As you probably read before on this page, MuseNN is a RNN, a Recurrent Neural Network. It's more precisely a Char-RNN (which works with characters). The concept is simple : it is trained on a text and tries to generate a similar text.

<big>Characters in music ?</big> Well, I knew the power of Char-RNN, and I was a bit too lazy to create my own RNN specially for music, so I was looking for a way to make music into text. And then I found the [ABC notation](http://abcnotation.com/). So I decided to convert [MIDI files](https://en.wikipedia.org/wiki/MIDI#MIDI_and_computers) into ABC notation and to train my Char-RNN.

<big>Isn't training a long task ?</big> If you already heard of RNN or Deep/Machine Learning, you may also have heard that it requires strong computers (that's why Google is pretty involved in Deep Learning...). So you may wonder how was I able to train a 152 neurons network (it is actually small for a RNN) from my $600 laptop in my room ?  
First, I used [Karpathy's Minimal Char-RNN](https://gist.github.com/karpathy/d4dee566867f8291f086) (in Python), which doesn't use heavy libraries like TensorFlow, so it's not hard to train it on a small device. Then, I didn't run it on my poor computer ![](http://photar.net/emoji/emoji-E056.png). I used my highschool's server (in France, you can do your studies in a high school), which is not a bad UNIX machine. It took around 4 hours for each file (hopefully I could train all 4 instruments of Moznn simultaneously).

<big>Can I train it on my own music ?</big> Sure ! As the code is mainly not mine, I made MuseNN open source !
