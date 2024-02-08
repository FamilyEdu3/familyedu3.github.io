from flask import Flask, render_template, jsonify
from flask import send_file, send_from_directory


import subprocess
import os, sys
import gradio as gr
from src.gradio_demo import SadTalker  
import numpy as np
from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
import platform

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.facerender.pirender_animate import AnimateFromCoeff_PIRender
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

import argparse
import sys
import re
import queue
from torch import no_grad, LongTensor
import logging
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import requests
import time
from pathlib import Path
import json
from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
import torch
from TTS.api import TTS
import broadscope_bailian
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from scipy.io.wavfile import write
import sys
import re
import queue
from torch import no_grad, LongTensor
import logging
import sounddevice as sd
from winsound import PlaySound
from vosk import Model, KaldiRecognizer
import requests
import time
from pathlib import Path
import json
import broadscope_bailian
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from moviepy.editor import VideoFileClip

import threading
import time

q = queue.Queue()
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def voice_input():

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="show list of audio devices and exit")
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        "-f", "--filename", type=str, metavar="FILENAME",
        help="audio file to store recording to")
    parser.add_argument(
        "-vd", "--voicedevice", type=int_or_str,
        help="input device (numeric ID or substring)")
    parser.add_argument(
        "-r", "--samplerate", type=int, help="sampling rate")
    parser.add_argument(
        "-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
    args = parser.parse_args(remaining)
    try:
        if args.samplerate is None:
            device_info = sd.query_devices(args.voicedevice, "input")
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info["default_samplerate"])

        if args.model is None:
            model = Model(lang="cn")
        else:
            model = Model(lang=args.model)

        if args.filename:
            dump_fn = open(args.filename, "wb")
        else:
            dump_fn = None
    except KeyboardInterrupt:
        print("\nDone")
        parser.exit(0)


    # print("You:")
    with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.voicedevice,
                           dtype="int16", channels=1, callback=callback):
        rec = KaldiRecognizer(model, args.samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                a = json.loads(rec.Result())
                a = str(a['text'])
                a = ''.join(a.split())
                if(len(a) > 0):
                    print(a)
                    user_input = a
                    return user_input
            if dump_fn is not None:
                dump_fn.write(data)

import broadscope_bailian

def bailian(input_prompt):
    access_key_id ="LTAI5tFtcHHovkJz8pHCRXC5"
    access_key_secret ="H3jbo73diqTVQKSIg7NMErVPoC0dRE"
    agent_key = "3f2927c4403245448b1891f8e92a99fb_p_efm"
    app_id ="0679f9ca800f4f2e996d2719cad8cd59"

    client = broadscope_bailian.AccessTokenClient(access_key_id=access_key_id, access_key_secret=access_key_secret,
                                                  agent_key=agent_key)
    token = client.get_token()

    prompt = input_prompt
    resp = broadscope_bailian.Completions(token=token).call(app_id=app_id, prompt=prompt)

    return resp

def process(choice):
    if choice == "快速生成":
        args.facerender='pirender'
        # args.ref_eyeblink="D:/Chat/ref/ref.mp4"
    elif choice == "优质生成":
        args.facerender='facevid2vid'


def clean_text(input_text):
    # Remove HTML tags and their contents
    text_without_html = re.sub(r'<.*?>', '', input_text)
    # Remove reference markers like [1], [2], etc.
    text_without_references = re.sub(r'\[\d+\]', '', text_without_html)
    return text_without_references


def save_file(file):
    path="D:/Chat/cloneaudio"
    if file is not None:
        with open(path, "wb") as f:
            f.write(file.read())
        return f"文件已保存到 {path}"
    return "没有文件上传"


app = Flask(__name__)
@app.route('/')
def index():
    # return render_template('web.html')
    return send_file('index.html')

@app.route('/generate', methods=['POST'])
def main():

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./audio.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./image/teacher.jpg', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=4,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true",default=True,help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 
    parser.add_argument("--facerender", default='facevid2vid', choices=['pirender', 'facevid2vid'] ) 
    

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()
    args.device="cuda"


    question = voice_input()
    resp = bailian(question)
    answer = clean_text(resp["Data"]["Text"]).replace('\n','')

    with open(os.path.join('.', 'answer.txt'), 'w', encoding='utf-8') as f:
        f.write(answer)
    # 将文本读取为字符串
    with open(os.path.join('.', 'answer.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
    tts.tts_to_file(text=text, speaker_wav="cloneaudio/vo_zhongli_friendship_03.wav", language="zh-cn", file_path="./audio.wav")

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, "output")
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    if args.facerender == 'facevid2vid':
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    elif args.facerender == 'pirender':
        animate_from_coeff = AnimateFromCoeff_PIRender(sadtalker_paths, device)
    else:
        raise(RuntimeError('Unknown model: {}'.format(args.facerender)))

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                            source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size, facemodel=args.facerender)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    
    shutil.move(result, save_dir+'.mp4')
    # print('The generated video is named:', save_dir+'.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)
    
    video_path=os.path.join(args.result_dir, "output.mp4")
    newvideo_path=os.path.join(args.result_dir, "newoutput.mp4")

    command = [
    'ffmpeg',
    '-i', video_path,  # 指定输入视频文件
    '-c:v', 'libx264',  # 设置视频编码器为libx264（H.264）
    '-preset', 'medium',  # 设置编码预设
    '-crf', '23',  # 设置恒定质量控制的质量因子
    '-c:a', 'aac',  # 设置音频编码器为AAC
    '-b:a', '128k',  # 设置音频比特率
    newvideo_path,  # 指定输出视频文件
    '-y'
    ]

    subprocess.run(command)


    

    # return jsonify({'videoPath': video_path})
    return jsonify(videoPath=newvideo_path)

@app.route('/results/<filename>')
def serve_results_file(filename):
    return send_from_directory('results', filename)



if __name__ == '__main__':
  

    # Get device
    device = "cuda"
    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    
    
    app.run(debug=True)


















