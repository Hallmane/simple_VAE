# building a VAE from scratch
using mnist 28x28 images to begin with 

## Future ideas
* is LSTFT fast enough to do some sort of live decoding from mel spectrogram to audio?
* VAE+GANs?

## TODO
* Build a sound generating VAE with this video 'https://www.youtube.com/watch?v=fWSoEqWNh8w'
* RAVE paper `https://arxiv.org/pdf/2111.05011.pdf`


# RAVE model
## encoder
"simple_cnn"(multiband decompositions)  -> 128-dim latent representation
More specifically: h_dims = [64, 128, 256, 512] with strides [4,4,4,2] -> 128 latent space

```
audio_in -> 
    multiband_decomposition ->
        N=4*( Conv1d -> BatchNorm1d -> LeakyReLU )
            --> Conv1d -> mean
            --> Conv1d -> softplus -> variance
```

## decoder
```
latent representation ->
    N=4*("upsampling layer" -> "residual stack")
        --> waveform Conv
            |--> (x)  -> wave*loud vector
        --> loudness Conv
        --> noise synthesizer + wave*loud vector -> audio out

```
where `upsampling layer` is:
```
-> leaky ReLU -> ConvTranspose1d
```
..., `residual stack` is (with a residual/skip connection before *3 block)
```
-> N=4*(*3( leaky ReLU -> Conv1d)) 
```
and `noise synthesizer` is:
```
->4*(Conv1d -> leaky ReLU) 
-> "white noise" 
    --> filter -> filter noise out

```






## What is...?
* multiband decomposition
* ConvTranspose1d
* the noise synthesizer
