import torch
from TTS.api import TTS
from logzero import logger
import logzero 

# print(logzero.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug(TTS().list_models())
tts = TTS("tts_models/en/ljspeech/vits").to(device)
tts.tts_to_file(text="Martha Baarbeero martos how are you my blood. If i punch you, you won't be able to tolerate it. You won't be able to sleep for 4 months.", file_path="/workspace/test.wav")