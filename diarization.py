from pyannote.audio import Pipeline
import torch
import torchaudio
import os

# Load diarization model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv('HF_TOKEN'))
pipeline.to(torch.device(device))

# Pre-loading audio files
file_path = 'media/1628082292/preprocessed_audio.wav'
waveform, sample_rate = torchaudio.load(file_path)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# run the pipeline on an audio file
diarization = pipeline(file_path)

class Diarization:
    def __init__(self, start, stop, speaker) -> None:
        self.start = start
        self.stop = stop
        self.speaker = speaker

check_list = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if len(check_list) == 0:
        check_list.append(Diarization(turn.start, turn.end, speaker))
    elif speaker != check_list[-1].speaker:
        check_list.append(Diarization(turn.start, turn.end, speaker))
    else:
        check_list[-1].stop = turn.end

postprocessing_list = []
for el in check_list:
    if el.stop - el.start > 0.5:
        if len(postprocessing_list) == 0:
            postprocessing_list.append(el)
        elif el.speaker == postprocessing_list[-1].speaker:
            postprocessing_list[-1].stop = el.stop
        else:
            postprocessing_list.append(el)


for el in postprocessing_list:
    print(f"start={el.start:.1f} stop={el.stop:.1f} {el.speaker}")
