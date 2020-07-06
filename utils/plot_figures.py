import argparse
import matplotlib.pyplot as plt
import os

from features import LogMelExtractor, calculate_logmel
import config

from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

font = 15

def plot_logmel(args):
    """Plot log Mel feature of one audio per class. 
    """

    # Arguments & parameters
    audios_dir = args.audios_dir
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    labels = config.labels
    
    # Paths
    audio_names = os.listdir(audios_dir)
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    #feature_list = []
    
    # Select one audio per class and extract feature

    chunk_length = 2000
    for audio_name in audio_names:
        if os.path.splitext(audio_name)[1] == '.wav':
            if not 'segment' in audio_name:
                audio = AudioSegment.from_wav(os.path.join(audios_dir, audio_name))

                chunks = make_chunks(audio, chunk_length)

                chunks[1].export(os.path.join(audios_dir, 'segment_'+audio_name), format='wav')


    for audio_name in audio_names:
        if os.path.splitext(audio_name)[1] == '.wav':
            if 'segment' in audio_name:

                audio_path = os.path.join(audios_dir, audio_name)

                feature = calculate_logmel(audio_path=audio_path,
                                           sample_rate=sample_rate,
                                           feature_extractor=feature_extractor)

                # feature_list.append(feature)
                # log mel spectrogram
                fig, axs = plt.subplots(1, 1)

                axs.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
                axs.set_xlabel('Time (s)', fontsize=font)
                axs.set_ylabel('Mel bins', fontsize=font)
                axs.xaxis.set_ticks([0, 31, 61])
                axs.yaxis.set_ticks([0, 32, 63])
                axs.xaxis.set_ticklabels(['0', '1', '2'], fontsize=font)
                axs.yaxis.set_ticklabels(['0', '32', '64'], fontsize=font)
                axs.xaxis.tick_bottom()

                axs.spines['top'].set_visible(False)
                axs.spines['right'].set_visible(False)
                plt.savefig(os.path.join(audios_dir, audio_name.split('.wav')[0] + '_logmel.pdf'))
                plt.close()


                # wave
                audio = AudioSegment.from_wav(os.path.join(audios_dir, audio_name))
                chunk = audio.get_array_of_samples()
                chunk = np.array(chunk)
                time = np.arange(0, 8000)
                fig, axs = plt.subplots(1, 1)
                axs.plot(time, chunk)
                axs.set_xlabel('Time (s)', fontsize=font)
                axs.xaxis.set_label_coords(1, 0.4)
                axs.set_ylabel('Amplitude', fontsize=font)
                axs.xaxis.set_ticks([0, 8000])
                axs.yaxis.set_ticks([])
                axs.xaxis.set_ticklabels(['0', '3'], fontsize=font)
                plt.xlim([0, 8000])
                axs.spines['bottom'].set_position(('data', 0))
                axs.spines['top'].set_visible(False)
                axs.spines['right'].set_visible(False)
                plt.savefig(os.path.join(audios_dir, audio_name.split('.wav')[0] + '_wave.pdf'))
                plt.close()

                # together
                fig, axs = plt.subplots(2, 1)
                axs[0].plot(time, chunk)
                axs[0].set_ylabel('Amplitude', fontsize=font)
                axs[0].xaxis.set_ticks([])
                axs[0].yaxis.set_ticks([])
                axs[0].set_xlim([0, 8000])
                axs[0].spines['bottom'].set_position(('data', 0))
                axs[0].spines['bottom'].set_color('gray')
                axs[0].spines['top'].set_visible(False)
                axs[0].spines['right'].set_visible(False)

                axs[1].matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
                axs[1].set_xlabel('Time (s)', fontsize=font)
                axs[1].set_ylabel('Mel bins', fontsize=font)
                axs[1].xaxis.set_ticks([0, 31, 61])
                axs[1].yaxis.set_ticks([0, 32, 63])
                axs[1].xaxis.set_ticklabels(['0', '1', '2'], fontsize=font)
                axs[1].yaxis.set_ticklabels(['0', '32', '64'], fontsize=font)
                axs[1].xaxis.tick_bottom()
                axs[1].spines['top'].set_visible(False)
                axs[1].spines['right'].set_visible(False)

                if '0008' in audio_name:
                    axs[1].axvline(x=2.5, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(3.5, 128, 'S1', fontsize=font-2)

                    axs[1].axvline(x=7, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(7.5, 128, 'stole', fontsize=font-2)

                    axs[1].axvline(x=12.5, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(13.2, 128, 'S2', fontsize=font-2)

                    axs[1].axvline(x=16, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(18, 128, 'diastole', fontsize=font-2)

                    axs[1].axvline(x=27, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(28.5, 128, 'S1', fontsize=font-2)

                    axs[1].axvline(x=32, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(32.2, 128, 'stole', fontsize=font-2)

                    axs[1].axvline(x=37, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(37.3, 128, 'S2', fontsize=font-2)

                    axs[1].axvline(x=40, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(41.5, 128, 'diastole', fontsize=font-2)

                    axs[1].axvline(x=50, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)
                    plt.text(51.5, 128, 'S1', fontsize=font-2)

                    axs[1].axvline(x=56, color='red', linestyle='--', ymax=2.05, lw=1, clip_on=False)


                plt.subplots_adjust(wspace=0, hspace=0.05)
                plt.savefig(os.path.join(audios_dir, audio_name.split('.wav')[0] + '_wave_logmel.pdf'))
                plt.close()



    '''
    # Plot
    rows_num = 3
    cols_num = 4
    n = 0
    
    fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 5))
    
    classes_num = len(labels)
    
    for n in range(classes_num):
        row = n // cols_num
        col = n % cols_num
        axs[row, col].matshow(feature_list[n].T, origin='lower', aspect='auto', cmap='jet')
        axs[row, col].set_title(labels[n])
        axs[row, col].set_ylabel('log mel')
        axs[row, col].yaxis.set_ticks([])
        axs[row, col].xaxis.set_ticks([0, seq_len])
        axs[row, col].xaxis.set_ticklabels(['0', '10 s'], fontsize='small')
        axs[row, col].xaxis.tick_bottom()
    
    for n in range(classes_num, rows_num * cols_num):
        row = n // cols_num
        col = n % cols_num
        axs[row, col].set_visible(False)
    
    fig.tight_layout()
    plt.show()
    
    '''
    
if __name__ == '__main__':
    audios_dir = "/home/zhao/NAS/data_work/Zhao/HSS2019/data/audio_plot"
    parser = argparse.ArgumentParser(description='Example of parser. ')

    parser.add_argument('--mode', type=str, default='plot_logmel')
    parser.add_argument('--audios_dir', type=str, default=audios_dir)

    args = parser.parse_args()

    if args.mode == 'plot_logmel':
        plot_logmel(args)

    else:
        raise Exception("Incorrect arguments!")



    '''
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot_logmel = subparsers.add_parser('plot_logmel')
    parser_plot_logmel.add_argument('--audios_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'plot_logmel':
        plot_logmel(args)
        
    else:
        raise Exception("Incorrect arguments!")
    '''