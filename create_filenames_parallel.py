import csv
import os
import random
import glob
import json
from tqdm import tqdm
import multiprocessing

wham_path = './wham_noise/tr'
librilight_path = './librilight/small'
savename = 'mix_5_spk_filenames_librilight_tr_small'
tag='small'
debug=False
max_duration=8
max_speaker=4 # only support less than 6 now
min_speaker=0

SEED=17
random.seed(SEED)

def process_one(i, tag, savename, spks, librilight_path, noise_files, debug):
    count=0
    os.makedirs(os.path.join('data', tag), exist_ok=True)
    file_path = os.path.join('data', tag, savename+ str(i)+'.csv')
    csvdata = [
        [
            "output_filename", "speaker_number", "noise_path", "noise_snr",
            "s1_path","s1_start","s1_end","s1_tag","s1_snr",
            "s2_path","s2_start","s2_end","s2_tag","s2_snr",
            "s3_path","s3_start","s3_end","s3_tag","s3_snr",
            "s4_path","s4_start","s4_end","s4_tag","s4_snr",
            "s5_path","s5_start","s5_end","s5_tag","s5_snr",
            ]
    ]
    for spk in spks:
        audiofiles = glob.glob(os.path.join(librilight_path, spk, '**/*.flac'), recursive=True)

        for audiofile in audiofiles:
            with open(audiofile.replace('.flac','.json'), 'r') as file:
                audiodata = json.load(file)
            vads = audiodata["voice_activity"]
            if debug:
                if count > 50:
                    break
            for vad in tqdm(vads):
                if float(vad[1]) - float(vad[0]) > 2.:
                    count+=1
                    copied_spks = spks[:]
                    copied_spks.remove(spk)
                    # print(len(copied_spks))
                    speaker_number = random.randint(min_speaker, max_speaker)
                    another_spks = random.sample(copied_spks, 4)
                    another_audiofiles1 = glob.glob(os.path.join(librilight_path, another_spks[0], '**/*.flac'), recursive=True)
                    another_audiofiles2 = glob.glob(os.path.join(librilight_path, another_spks[1], '**/*.flac'), recursive=True)
                    another_audiofiles3 = glob.glob(os.path.join(librilight_path, another_spks[2], '**/*.flac'), recursive=True)
                    another_audiofiles4 = glob.glob(os.path.join(librilight_path, another_spks[3], '**/*.flac'), recursive=True)
                    another_audiofile1 = random.choice(another_audiofiles1)
                    another_audiofile2 = random.choice(another_audiofiles2)
                    another_audiofile3 = random.choice(another_audiofiles3)
                    another_audiofile4 = random.choice(another_audiofiles4)
                    with open(another_audiofile1.replace('.flac','.json'), 'r') as file:
                        another_audiodata1 = json.load(file)
                    with open(another_audiofile2.replace('.flac','.json'), 'r') as file:
                        another_audiodata2 = json.load(file)
                    with open(another_audiofile3.replace('.flac','.json'), 'r') as file:
                        another_audiodata3 = json.load(file)
                    with open(another_audiofile4.replace('.flac','.json'), 'r') as file:
                        another_audiodata4 = json.load(file)
                    another_vads1 = another_audiodata1["voice_activity"]
                    another_vads2 = another_audiodata2["voice_activity"]
                    another_vads3 = another_audiodata3["voice_activity"]
                    another_vads4 = another_audiodata4["voice_activity"]
                    another_vad1 = random.choice(another_vads1)
                    another_vad2 = random.choice(another_vads2)
                    another_vad3 = random.choice(another_vads3)
                    another_vad4 = random.choice(another_vads4)
                    noisefile = random.choice(noise_files)
                    s1_start = random.uniform(0, max_duration-1)
                    s2_start = random.uniform(0, max_duration-1)
                    s3_start = random.uniform(0, max_duration-1)
                    s4_start = random.uniform(0, max_duration-1)
                    s5_start = random.uniform(0, max_duration-1)
                    s1_snr = random.uniform(-3, 6)
                    s2_snr = random.uniform(-3, 6)
                    s3_snr = random.uniform(-3, 6)
                    s4_snr = random.uniform(-3, 6)
                    s5_snr = random.uniform(-3, 6)
                    noise_snr = random.uniform(-6, 3)
                    csvdata.append([
                        str(count)+'.wav',
                        speaker_number,
                        noisefile,
                        noise_snr,
                        audiofile,
                        vad[0],
                        vad[1],
                        s1_start,
                        s1_snr,
                        another_audiofile1,
                        another_vad1[0],
                        another_vad1[1],
                        s2_start,
                        s2_snr,
                        another_audiofile2,
                        another_vad2[0],
                        another_vad2[1],
                        s3_start,
                        s3_snr,
                        another_audiofile3,
                        another_vad3[0],
                        another_vad3[1],
                        s4_start,
                        s4_snr,
                        another_audiofile4,
                        another_vad4[0],
                        another_vad4[1],
                        s5_start,
                        s5_snr,
                        ])
            
        
    # Writing to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csvdata)
    print(f'Data has been written to {file_path}')


# Reading all noise files
noise_files = [os.path.abspath(os.path.join(wham_path, file)) for file in os.listdir(wham_path) if file.endswith('.wav')]
spks = [name for name in os.listdir(librilight_path) if os.path.isdir(os.path.join(librilight_path, name)) and not os.path.isfile(os.path.join(librilight_path, name))]
num_subspks = 50
subspks_size = len(spks) // num_subspks
random.shuffle(spks)
# Split the list into subspks
subspks = [spks[i*subspks_size:(i+1)*subspks_size] for i in range(num_subspks)] 

# If there are remaining elements, distribute them among the subspks
if len(spks) % num_subspks != 0:
    remainder = len(spks) % num_subspks
    for i in range(remainder):
        subspks[i].append(spks[-(i+1)])
        spks.pop()      
        
cmds = []

for i in range(len(subspks)):
    cmds.append((i, tag, savename, subspks[i], librilight_path, noise_files, debug))
    
random.shuffle(cmds)
with multiprocessing.Pool(processes=50) as pool:
    pool.starmap(process_one, cmds)
