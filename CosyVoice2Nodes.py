import os
import sys
import torch
import torchaudio
import random
import librosa
import gc
import numpy as np
from tqdm import tqdm
from typing import Generator
import folder_paths
import hashlib
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
# sys.path.append(os.path.join(now_dir, 'cosyvoice/cosyvoice'))
sys.path.append(os.path.join(now_dir, 'third_party/Matcha-TTS'))


from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed
from comfy_extras.nodes_audio import LoadAudio



prompt_sample_rate=16000
target_sample_rate=24000
max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sample_rate * 0.2))], dim=1)
    return speech

def audio_resample(waveform, source_sr):
    waveform = waveform.squeeze(0)
    speech = waveform.mean(dim=0,keepdim=True)
    if source_sr != prompt_sample_rate:
        speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sample_rate)(speech)
    return speech

def build_model_input(model:CosyVoice2,prompt_audio,prompt_text):
    waveform = prompt_audio['waveform']
    sample_rate = prompt_audio['sample_rate']
    speech  = audio_resample(waveform, sample_rate)
    prompt_speech_16k = postprocess(speech)

    speech_token, speech_token_len = model.frontend._extract_speech_token(prompt_speech_16k)
    # if save_as_speaker:
    #     if save_speaker_name is None or len(save_speaker_name) == 0:
    #         raise Exception("The save_speaker_name is required!")
    #     if not replace_exist_speaker and save_speaker_name in model.frontend.spk2info:
    #         raise Exception("Speaker with the name "+save_speaker_name+" does exist, please check the name.")
            

    prompt_text = model.frontend.text_normalize(prompt_text, split=False,text_frontend=True)
    
    prompt_text_token, prompt_text_token_len = model.frontend._extract_text_token(prompt_text)

    prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=model.sample_rate)(prompt_speech_16k)
    speech_feat, speech_feat_len = model.frontend._extract_speech_feat(prompt_speech_resample)
    if model.sample_rate == 24000:
        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
        speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len

    embedding = model.frontend._extract_spk_embedding(prompt_speech_16k)
    model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                'llm_embedding': embedding, 'flow_embedding': embedding}
    return model_input
 
class CosyVoice2ZeroShot:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "run"

    OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "CosyVoice/V2"

    
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "model": ("MODEL_CosyVoice2", { "tooltip": "This is a model white CosyVoice2"}),
                "tts_text": ("STRING",{"multiline":True}),
                "save_as_speaker":("BOOLEAN",{"default":False}),
                "save_speaker_name":("STRING",),
                "replace_exist_speaker":("BOOLEAN",{"default":False}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2, "step": 0.1}),
                "seed":("INT",{
                    "default": 0
                }),
               
            },
            "optional":{
                "prompt_audio": ("AUDIO",),
                "prompt_text": ("STRING",{"multiline":True}),
                "select_speaker": ("STRING",{"default":""})
            }
        }
    def get_output_data(self,generator):
        # output_list = []
        # for out_dict in generator:
        #     output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768 
        #     output_numpy = output_numpy.astype(np.int16)
        #     output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
        # return torch.cat(output_list,dim=1).unsqueeze(0)
        
        chunks = []
        for chunk in generator:
            chunks.append(chunk['tts_speech'].numpy().flatten())
        output = np.array(chunks)
        return torch.from_numpy(output).unsqueeze(0)

    def run_model(self, model:CosyVoice2,tts_text=None,save_as_speaker=False,save_speaker_name=None,replace_exist_speaker=False,speed=1.0,seed=0,prompt_audio=None, prompt_text=None,select_speaker=""):
        print('开始推理',prompt_text,prompt_audio,tts_text)
        if tts_text is None or len(tts_text) == 0:
            raise Exception("The tts_text is required!")
        
        if prompt_audio is not None and (prompt_text is None or len(prompt_text) == 0):
            raise Exception("The prompt_text is required!")
       
        if prompt_audio is None and len(select_speaker)==0:
            raise Exception("Parameters prompt_audio, select_speaker, need to select one!")

        set_all_random_seed(seed)

        if prompt_audio is not None:
            # waveform = prompt_audio['waveform']
            # sample_rate = prompt_audio['sample_rate']
            # speech  = audio_resample(waveform, sample_rate)
            # prompt_speech_16k = postprocess(speech)
            # print('model.frontend.spk2info',model.frontend.spk2info)
            # speech_token, speech_token_len = model.frontend._extract_speech_token(prompt_speech_16k)
            if save_as_speaker:
                if save_speaker_name is None or len(save_speaker_name) == 0:
                    raise Exception("The save_speaker_name is required!")
                if not replace_exist_speaker and save_speaker_name in model.frontend.spk2info:
                    raise Exception("Speaker with the name "+save_speaker_name+" does exist, please check the name.")
                    

            # prompt_text = model.frontend.text_normalize(prompt_text, split=False,text_frontend=True)
            
            # prompt_text_token, prompt_text_token_len = model.frontend._extract_text_token(prompt_text)

            # prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=model.sample_rate)(prompt_speech_16k)
            # speech_feat, speech_feat_len = model.frontend._extract_speech_feat(prompt_speech_resample)
            # if model.sample_rate == 24000:
            #     token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            #     speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            #     speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len

            # embedding = model.frontend._extract_spk_embedding(prompt_speech_16k)
            # model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
            #             'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
            #             'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
            #             'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
            #             'llm_embedding': embedding, 'flow_embedding': embedding}
            model_input=build_model_input(model,prompt_audio,prompt_text)
            
            if prompt_audio is not None and save_as_speaker:
                model.frontend.spk2info[save_speaker_name]=model_input
                model.save_spkinfo()
        else:
            model_input=model.frontend.spk2info[select_speaker]
        
        for i in tqdm(model.frontend.text_normalize(tts_text, split=True, text_frontend=True)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
                
            tts_text_token, tts_text_token_len = model.frontend._extract_text_token(i)
            model_input['text']=tts_text_token
            model_input['text_len']=tts_text_token_len
            
            for model_output in model.model.tts(**model_input, stream=False, speed=speed):
                yield model_output


    def run(self, model:CosyVoice2,tts_text="",save_as_speaker=False,save_speaker_name="",replace_exist_speaker=False,speed=1.0,seed=123,prompt_audio=None, prompt_text=None,select_speaker="None"):
        generator=self.run_model(model,tts_text,save_as_speaker,save_speaker_name,replace_exist_speaker,speed,seed,prompt_audio, prompt_text,select_speaker)
        audio=self.get_output_data(generator)
        
        return ({"waveform": audio, "sample_rate":model.sample_rate},)

    
class CosyVoice2CreateSpeaker():
    RETURN_TYPES = ("Speakers",)
    RETURN_NAMES = ("speakers",)

    FUNCTION = "run"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "CosyVoice/V2"
    


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice2", { "tooltip": "This is a model white CosyVoice2"}),
                "prompt_audio": ("AUDIO",),
                "prompt_text": ("STRING",{"multiline":True}),
                "save_speaker_name":("STRING",),
                "replace_exist_speaker":("BOOLEAN",{"default":False})
            },
        }


    def run(self,model:CosyVoice2,save_speaker_name,replace_exist_speaker,prompt_audio, prompt_text):
        if prompt_audio is None:
            raise Exception("The prompt_audio is required!")
        if prompt_text is None or len(prompt_text) == 0:
            raise Exception("The prompt_text is required!")
        if save_speaker_name is None or len(save_speaker_name) == 0:
            raise Exception("The save_speaker_name is required!")
        if not replace_exist_speaker and save_speaker_name in model.frontend.spk2info:
            raise Exception("Speaker with the name "+save_speaker_name+" does exist, please check the name.")
        model_input=build_model_input(model,prompt_audio,prompt_text)
        model.frontend.spk2info[save_speaker_name]=model_input
        model.save_spkinfo()
        return (model.list_available_spks(),)

class CosyVoice2Loader():
    RETURN_TYPES = ("MODEL_CosyVoice2",)
    RETURN_NAMES = ("model",)

    FUNCTION = "run"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "CosyVoice/V2"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "auto_download": ("BOOLEAN", {"default": True},),
                "load_jit": ("BOOLEAN", {"default": False},),
                "load_vllm": ("BOOLEAN", {"default": False},),
                "load_trt": ("BOOLEAN", {"default": False},),
                "fp16": ("BOOLEAN", {"default": False},),
            },
        }



    def run(self,auto_download=True,load_jit=True, load_vllm=False,load_trt=False,fp16=False):
        model_dir_root=os.path.join(folder_paths.models_dir,"CosyVoice")
        model_dir = os.path.join(model_dir_root,"CosyVoice2-0.5B")
        if auto_download:
            if not os.path.exists(model_dir):
                print("download.......CosyVoice")
                from modelscope import snapshot_download
                snapshot_download('iic/CosyVoice2-0.5B', local_dir= model_dir)
                snapshot_download('iic/CosyVoice-ttsfrd', local_dir=os.path.join(model_dir_root,'CosyVoice-ttsfrd'))
                os.system(f'cd {model_dir_root}/CosyVoice-ttsfrd/ && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && apt install -y unzip && unzip resource.zip -d .')
        if not os.path.exists(model_dir):
            raise Exception("The model is not found, please check the model path.")
        cosyVoice2=CosyVoice2(model_dir, load_jit=load_jit, load_trt=load_trt, load_vllm=load_vllm,fp16=fp16)
        return (cosyVoice2,)
    
class CosyVoice2SpeakerList():
    RETURN_TYPES = ("Speakers",)
    RETURN_NAMES = ("speakers",)

    FUNCTION = "run"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "CosyVoice/V2"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice2", { "tooltip": "This is a model white CosyVoice2"}),
            },
        }

    def run(self,model:CosyVoice2):
        return (model.list_available_spks(),)
  
NODE_CLASS_MAPPINGS = {
    "CosyVoice2ZeroShot": CosyVoice2ZeroShot,
    "CosyVoice2Loader":CosyVoice2Loader,
    "CosyVoice2CreateSpeaker":CosyVoice2CreateSpeaker,
    "CosyVoice2SpeakerList":CosyVoice2SpeakerList
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoice2ZeroShot": "CosyVoice2 Zero Shot",
    "CosyVoice2Loader": "CosyVoice2 Model Loader",
    "CosyVoice2CreateSpeaker":"Create CosyVoice2 Speaker",
    "CosyVoice2SpeakerList":"CosyVoice2 Speaker List"
}
