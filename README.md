# Generating the LibriLight-Mix dataset

This script supports generating noisy multiple-speaker mixture audio for training with the Libri-Light dataset, which can be served as training materials for large-scale noisy speech separation, voice activity detection, overlapping speech detection, speaker counting and speaker diarization.

If you want to add reverb, please refer to [LibriLightMix-WHAMR](https://github.com/WangHelin1997/LibriLightMix-WHAMR).

## Python requirements

Requires python 3.8, and the numpy, scipy, and pandas packages
```sh
$ pip install -r requirements.txt
```

## Prerequisites

This requires the [Libri-Light](https://github.com/facebookresearch/libri-light) dataset,
and the [WHAM](http://wham.whisper.ai/) noise corpus.

## Creating LibriLight-Mix

### Creating meta files

```sh
$ python create_filenames.py 
```
Change the following arguments in the script:
* **wham_path**:  Folder where the unzipped wham_noise was downloaded (training set).
* **librilight_path**: Folder where the unzipped Libri-Light data was downloaded.
* **debug**: Whether to process a dummy dataset.
* **max_duration**: The duration of audio file to simulate.
* **min_speaker**: The minimum number of speakers within the mixture audio.
* **max_speaker**: The maximum number of speakers within the mixture audio.

The simulated mixture audio will randomly sample a speaker number from **<min_speaker>** to **<max_speaker>**.

### Creating mixture files

```sh
$ python create_wham_from_scratch.py \
    --output-dir ./librilight_whamr/ \
    --sr 16000 \
    --fixed-len 8
 
```

The arguments for the script are:
* **output-dir**: Where to write the new dataset.
* **sr**: Sampling rate.
* **fixed-len**: The duration of audio file to simulate. Set the same as **<max_duration>** in **<create_filenames.py>**.

## Creating LibriLight-Mix parallelly with mulitple CPUs

### Creating meta files

```sh
$ python create_filenames_parallel.py 
```
Change the following arguments in the script:
* **wham_path**:  Folder where the unzipped wham_noise was downloaded (training set).
* **librilight_path**: Folder where the unzipped Libri-Light data was downloaded.
* **savename**: Name of the meta .csv file to save.
* **tag**: Name of the meta .csv folder to save.
* **debug**: Whether to process a dummy dataset.
* **max_duration**: The duration of audio file to simulate.
* **min_speaker**: The minimum number of speakers within the mixture audio.
* **max_speaker**: The maximum number of speakers within the mixture audio.

### Creating mixture files

```
for i in $(seq 0 49)
do
    python create_wham_from_scratch_parallel.py \
        --output-dir "./LibrilightMix-small/$i/" \
        --filepath "data/small/mix_5_spk_filenames_librilight_tr_small$i.csv" \
        --sr 16000 \
        --fixed-len 8
done
```

The arguments for the script are:
* **output-dir**: Where to write the new dataset.
* **filepath**: Name of the saved meta .csv folder.
* **sr**: Sampling rate.
* **fixed-len**: The duration of audio file to simulate. Set the same as **<max_duration>** in **<create_filenames_parallel.py>**.



## Output data organization

For each utterance in the training (tr) set folder, the following wav files are written:

1. noise: contains the isolated background noise from WHAM!

2. s1: isolated data from speaker 1

3. s2: isolated data from speaker 2

4. s3: isolated data from speaker 3

5. s4: isolated data from speaker 4

6. s5: isolated data from speaker 5

7. mix_clean: clean speech separation for N speakers, contains mixture of s1, s2, ..., sN.

8. mix_noisy: noisy speech separation for N speakers, contains mixture of s1, s2, ..., sN, and noise.


## Reference

https://wham.whisper.ai/WHAMR_README.html
