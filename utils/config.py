#sample_rate = 44100
#window_size = 1024 # 1024 # 2048
#overlap = 256 # 256 # 672   # So that there are 320 frames in an audio clip
#seq_len = 344 # 573 # 320
#mel_bins = 64

sample_rate = 16000
window_size = 512 #1024 
overlap = 256 #256 
seq_len = 373 #344 
mel_bins = 64 #64 #257

labels = ['anger', 'disgust', 'fear', 'guilt', 'happiness', 'sadness', 'surprise']

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}
