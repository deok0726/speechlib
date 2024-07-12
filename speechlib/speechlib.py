from core_analysis import (core_analysis)
from re_encode import (re_encode)
from convert_to_mono import (convert_to_mono)
from convert_to_wav import (convert_to_wav)

class Transcriptor:

    def __init__(self, file, log_folder, language, modelSize, ACCESS_TOKEN, voices_folder=None, quantization=False):
        '''transcribe a wav file 
        
            arguments:

            file: name of wav file with extension ex: file.wav

            log_folder: name of folder where transcript will be stored

            language: language of wav file

            modelSize: tiny, small, medium, large, large-v1, large-v2, large-v3 (bigger model is more accurate but slow!!)

            voices_folder: folder containing subfolders named after each speaker with speaker voice samples in them. This will be used for speaker recognition

            quantization: whether to use int8 quantization or not (default=False)

            see documentation: https://github.com/Navodplayer1/speechlib
        '''
        self.file = file
        self.voices_folder = voices_folder
        self.language = language
        self.log_folder = log_folder
        self.modelSize = modelSize
        self.quantization = quantization
        self.ACCESS_TOKEN = ACCESS_TOKEN

    def whisper(self):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "whisper", self.quantization)
        return res
    
    def faster_whisper(self):
        res = core_analysis(self.file, self.voices_folder, self.log_folder, self.language, self.modelSize, self.ACCESS_TOKEN, "faster-whisper", self.quantization)
        return res

class PreProcessor:
    '''
    class for preprocessing audio files.

    methods:

    re_encode(file) -> re-encode file to 16-bit PCM encoding  

    convert_to_mono(file) -> convert file from stereo to mono  

    mp3_to_wav(file) -> convert mp3 file to wav format  

    '''

    def re_encode(self, file):
        re_encode(file)
    
    def convert_to_mono(self, file):
        convert_to_mono(file)

    def convert_to_wav(self, file):
        path = convert_to_wav(file)
        return path
