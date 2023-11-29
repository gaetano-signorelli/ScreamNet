# ScreamNet

This project aims at creating an AI system, based on a GAN neural network, capable of transforming vocal audio inputs by applying a screaming effect that turns voices (whether they come from songs, speeches or even whispers) into realistic screams and growls, to the delight of metal lovers 🤘🎵.<br>
This allows, for example, to reinterpret songs by generating an original screaming counterpart, or to experiment with personal recordings to produce metal-like vocals.

## How it works

The implementation is based on the **MelGAN** architecture [^1], which has been revised in a couple of points that will be briefly discussed:

1. Differently from the original paper, whose goal is to produce a raw waveform starting from a mel-spectrogram (output of another neural model) for a Text-To-Speech task, the designed model takes a raw waveform in input. Nonetheless, it is immediately transformed into the corresponding mel-spectrogram also in this case: this happens inside of the network, which is an end-to-end model, that also handles preprocessing operations such as the normalization, without directly relying on external libraries as librosa, making the model completely independent and portable (see the "App" section below). Similarly to the real MelGAN, this input is made up a 1 second of audio, processed at 22050Hz.

2. The hinge loss has been preserved, as well as the general structure (including the use of three different discriminators operating at different levels), but the feature matching loss has been suppressed and replaced with a more adeguate loss for this task: a correlation loss, which ensures that the recreated signals not only resemble a screaming sound (like enforced by the GAN approach) but that they also match the original audios. This has been carried out by bringing the original and artificial waveforms into a mel-spectrogram representation (operating in a space that gives more precision in reproducing sound fidelity, like repeating the exact same original words) and the maximizing their cosine similarity (thus their correlation).

## Results

The model needs to be trained on two datasets: one consisting of the audio files the network will have to transform (diversified in genres, voices and styles); the other made up of screams (diversified in terms of vocal techniques) coming from real metal songs.<br>
The first dataset can be found inside *data/Dataset/Vocals* and it is a preprocessed version of the well-known GTZAN dataset [^2]: vocal tracks have been separated from instruments by means of an external AI (there are many free options online, such as this one: https://github.com/stemrollerapp/stemroller) and silence has been removed.<br>
This network has been trained for 100k steps, using a batch size of 8. Other training parameters can be found inside the *src/config.py* module.

Here are some samples:

### Original tracks

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/1fe0611e-31a8-4bca-98aa-cad3f6a771d6

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/802ffbd3-bae7-4730-88f6-ef7d76387d09

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/b0db447b-c5ea-4ed5-ad84-ad11a7c0dfe1

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/6261b786-51a8-4266-b0af-645355a53de6

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/82551d37-9e86-4ff7-9ebb-d3f1558a0dfa

### ScreamNet tracks

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/e12b287b-f5f4-491f-8444-9d1aec07fd6f

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/70bebcc7-7e3c-45d8-8391-25e566a5007a

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/57028546-8997-4c1e-b9c4-4a4a4af33776

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/aca22a6d-3411-4e98-8456-75562134149e

https://github.com/gaetano-signorelli/ScreamNet/assets/51027023/2213fd3a-5d28-47a1-a575-bd70e26a380c

These results show a good quality in the screaming effect, with little to no imperfections (e.g., no distortions or missing words), depending on the original voice. It can also be noticed how the model operates differently adjusting the screaming techique to fit vocals at best, taking into account factors such as intonation, rythm and timbre.

More samples can be found [here](https://drive.google.com/drive/u/0/folders/1szeo8SHQeCP30tEvfruL3ZJFaEkc2fBw), where vocals have been merged back to instrumental parts, effectively giving rise to metal covers of real songs.

[^1]: https://arxiv.org/abs/1910.06711
[^2]: https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf
