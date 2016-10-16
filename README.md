# MuseNN - When RNN meets music

### Summary :

1.  [Some compositions](#comp)
2.  [How does it work](#func)
3.  [Contact MuseNN](#contact)

## Some composition

MuseNN was trained on different styles. The best results came from House melodies on piano and a Classical quatuor from Mozart.

The other styles I trained the RNN on resulted on complete mess of notes (almost like [Black MIDIs](https://www.youtube.com/watch?v=I906a5msynw)), or were too close from the training models. So here are the best I could get (training on little data - around 250 chars -, with a minimal RNN, on my highschool's server) :

| ► NNHouse | 1:42 | 1 instrument (piano) | Trained on ten 15-seconds melodies |
| ► Moznn | 4:43 | 4 instruments (violin 1 and 2, viola, cello) | Trained on Mozart's [String quartet nÂ°1 in G](https://www.youtube.com/watch?v=2I9LyiGrclc) |

### Downloads :

MP3s :

*   [NNHouse (1.95MB)](http://www.mediafire.com/file/nkc82sz6y70mclt/NNHouse+-+Final.mp3)
*   [Moznn (5.41MB)](http://www.mediafire.com/file/c0bfuctnwrn5psc/MOZNN.mp3)

Scores :

*   [NNHouse (78KB)](http://www.mediafire.com/file/1c4bh5sc2vvbqdh/NNHOUSE+-+Final.pdf)
*   [Moznn (136KB)](http://www.mediafire.com/file/k94g4wgb97h7cyj/MOZNN.pdf)

## How does it work ?

First things first : <big>What is MuseNN?</big> As you probably read before on this page, MuseNN is a RNN, a Recurrent Neural Network. It's more precisely a Char-RNN (which works with characters). The concept is simple : it is trained on a text and tries to generate a similar text.

<big>Characters in music?</big> Well, I knew the power of Char-RNN, and I was a bit too lazy to create my own RNN specially for music, so I was looking for a way to make music into text. And then I found the [ABC notation](http://abcnotation.com/). So I decided to convert [MIDI files](https://en.wikipedia.org/wiki/MIDI#MIDI_and_computers) into ABC notation and to train my Char-RNN.

<big>Isn't training a long task?</big> If you already heard of RNN or Deep/Machine Learning, you may also have heard that it requires strong computers (that's why Google is pretty involved in Deep Learning...). So you may wonder how was I able to train a 152 neurons network (it is actually small for a RNN) from my $600 laptop in my room ?
First, I used [Karpathy's Minimal Char-RNN](https://gist.github.com/karpathy/d4dee566867f8291f086) (in Python), which doesn't use heavy libraries like TensorFlow, so it's not hard to train it on a small device. Then, I didn't run it on my poor computer ![](http://photar.net/emoji/emoji-E056.png). I used my highschool's server (in France, you can do your studies in a high school), which is not a bad UNIX machine. It took around 4 hours for each file (hopefully I could train all 4 instruments of Moznn simultaneously).

<big>Can I train it on my own music?</big> Sure! As the code is mainly not mine, I made [MuseNN open source](https://github.com/p4ulolol/musenn)! You just have to clone the Git repository to your files. Then, here is how to train it :

1.  Find some good MIDI files (the more the better!) and convert them to ABC (I used [EasyABC](https://sourceforge.net/projects/easyabc/) to do this).
2.  Copy <u>only the notes lines</u> (not the headers like "X:...", "K:..." or "V:...") and paste it in a text file in MuseNN's folder.
3.  Now you can train MuseNN like this:

    ```musenn.py -f your_file.txt -o output_folder -m 2e6```

4.  After 2 million iterations, it should have generated many files, and one prefixed with "FINAL". You can copy its content in a .abc file or directly paste it in EasyABC.
5.  Then you'll have to correct several mistakes (depending on your computers' quality, and the loss of the file), but hopefully, EasyABC tells where they are (just ignore "Bad tie" and "Line under/overfull")

And... You're done ! You can now export it as MIDI so you can listen the creation.

## Contact MuseNN

You can contact MuseNN (actually its creator, me, Paul Forveille, also known as [Cyxo](http://cyxo.cf/)) by the following ways :

*   By mail : [MuseNN@Cyxo.33mail.com](mailto:MuseNN@Cyxo.33mail.com)
*   On Twitter : [@cyxo_officiel](https://twitter.com/cyxo_officiel)

I'm French so if you want to speak French, you can ![](http://forums.everythingicafe.com/data/MetaMirrorCache/photar.net_emoji_emoji_E405.png)

Je suis français !

_Enjoy!_
