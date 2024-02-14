from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import cv2
from pyiqa import create_metric
import torch.nn.functional as F
from math import log10, sqrt
import numpy as np

niqe = create_metric("niqe", metric_mode="NR")

os.environ['CURL_CA_BUNDLE'] = ''

def get_model_info(model_ID, device):
    model = CLIPModel.from_pretrained(model_ID).to(device)
    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    return model, processor, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"
model, processor, tokenizer = get_model_info(model_ID, device)

def get_single_text_embedding(text): 
    inputs = tokenizer(text, return_tensors = "pt")
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    text_embeddings = model.get_text_features(**inputs)
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np

def get_single_image_embedding(my_image):
    image = processor(
    text = None,
    images = my_image,
    return_tensors="pt"
    )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    # convert the embeddings to numpy array
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np


original_cams = []
cams = []
texts = []



original_cams.append(cv2.VideoCapture("renders/face/2023-07-05_135853_face.mp4"))
cams.append(cv2.VideoCapture("renders/face/2023-06-29_132836_bronze_2444_iters_debugged_heatmpa_field_bronze_ours.mp4"))
texts.append("his face is a bronze statue")

original_cams.append(cv2.VideoCapture("renders/face/2023-07-05_135853_face.mp4"))
cams.append(cv2.VideoCapture("renders/face/2023-07-04_142825_einstein_ouurs.mp4"))
texts.append("Albert einstein")

original_cams.append(cv2.VideoCapture("renders/face/2023-07-05_135853_face.mp4"))
cams.append(cv2.VideoCapture("renders/face/2023-07-04_142830_joker_ours.mp4"))
texts.append("Heath Ledger's joker")

original_cams.append(cv2.VideoCapture("renders/face/2023-07-05_135853_face.mp4"))
cams.append(cv2.VideoCapture("renders/face/2023-07-04_143758_musstache_ours.mp4"))
texts.append("A man with a mustache")

original_cams.append(cv2.VideoCapture("renders/fern/2023-07-05_140416_fern.mp4"))
cams.append(cv2.VideoCapture("renders/fern/2023-07-04_155052_ice_ours.mp4"))
texts.append("An ice statue of a plant")

original_cams.append(cv2.VideoCapture("renders/fern/2023-07-05_140416_fern.mp4"))
cams.append(cv2.VideoCapture("renders/fern/2023-07-04_155055_fire_ours.mp4"))
texts.append("A scene on fire")

original_cams.append(cv2.VideoCapture("renders/bear/2023-07-05_135759_bear.mp4"))
cams.append(cv2.VideoCapture("renders/bear/2023-07-05_095251_panda_ours.mp4"))
texts.append("A Panda")

original_cams.append(cv2.VideoCapture("renders/bear/2023-07-05_135759_bear.mp4"))
cams.append(cv2.VideoCapture("renders/bear/2023-07-05_095252_grizzly_ours.mp4"))
texts.append("A Grizzly bear")

original_cams.append(cv2.VideoCapture("renders/bear/2023-07-05_135759_bear.mp4"))
cams.append(cv2.VideoCapture("renders/bear/2023-07-05_095253_polar_ours.mp4"))
texts.append("A polar bear")

original_cams.append(cv2.VideoCapture("renders/fangzhou-small/2023-07-05_135627_fangzhou.mp4"))
cams.append(cv2.VideoCapture("renders/fangzhou-small/2023-07-05_115157_elf_ours.mp4"))
texts.append("A Tolkien Elf")

original_cams.append(cv2.VideoCapture("renders/fangzhou-small/2023-07-05_135627_fangzhou.mp4"))
cams.append(cv2.VideoCapture("renders/fangzhou-small/2023-07-05_133123_blue_ours.mp4"))
texts.append("A man with blue hair")

original_cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_113439_farm.mp4"))
cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_105954_snow_ours.mp4"))
texts.append("Snow")

original_cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_113439_farm.mp4"))
cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_105954_sunset_ours.mp4"))
texts.append("sunset")

original_cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_113439_farm.mp4"))
cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_105954_snow_ours.mp4"))
texts.append("storm")




original_cams.append(cv2.VideoCapture("renders/face/2023-07-05_135853_face.mp4"))
cams.append(cv2.VideoCapture("renders/face/2023-07-04_140258_bronze_in2n.mp4"))
texts.append("his face is a bronze statue")

original_cams.append(cv2.VideoCapture("renders/face/2023-07-05_135853_face.mp4"))
cams.append(cv2.VideoCapture("renders/face/2023-07-04_133633_einstein_in2n.mp4"))
texts.append("Albert einstein")

original_cams.append(cv2.VideoCapture("renders/face/2023-07-05_135853_face.mp4"))
cams.append(cv2.VideoCapture("renders/face/2023-07-04_133703_joker_in2n.mp4"))
texts.append("Heath Ledger's joker")

original_cams.append(cv2.VideoCapture("renders/face/2023-07-05_135853_face.mp4"))
cams.append(cv2.VideoCapture("renders/face/2023-07-04_133650_mustache_in2n.mp4"))
texts.append("A man with a mustache")

original_cams.append(cv2.VideoCapture("renders/fern/2023-07-05_140416_fern.mp4"))
cams.append(cv2.VideoCapture("renders/fern/2023-07-04_162329_ice_in2n.mp4"))
texts.append("An ice statue of a plant")

original_cams.append(cv2.VideoCapture("renders/fern/2023-07-05_140416_fern.mp4"))
cams.append(cv2.VideoCapture("renders/fern/2023-07-04_162214_fire_in2n.mp4"))
texts.append("A scene on fire")

original_cams.append(cv2.VideoCapture("renders/bear/2023-07-05_135759_bear.mp4"))
cams.append(cv2.VideoCapture("renders/bear/2023-07-05_103557_panda_in2n.mp4"))
texts.append("A Panda")

original_cams.append(cv2.VideoCapture("renders/bear/2023-07-05_135759_bear.mp4"))
cams.append(cv2.VideoCapture("renders/bear/2023-07-05_103657_grizzly_in2n.mp4"))
texts.append("A Grizzly bear")

original_cams.append(cv2.VideoCapture("renders/bear/2023-07-05_135759_bear.mp4"))
cams.append(cv2.VideoCapture("renders/bear/2023-07-05_103700_polar_in2n.mp4"))
texts.append("A polar bear")

original_cams.append(cv2.VideoCapture("renders/fangzhou-small/2023-07-05_135627_fangzhou.mp4"))
cams.append(cv2.VideoCapture("renders/fangzhou-small/2023-07-05_122324_elf_in2n.mp4"))
texts.append("A Tolkien Elf")

original_cams.append(cv2.VideoCapture("renders/fangzhou-small/2023-07-05_135627_fangzhou.mp4"))
cams.append(cv2.VideoCapture("renders/fangzhou-small/2023-07-05_122432_blue_in2n.mp4"))
texts.append("A man with blue hair")

original_cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_113439_farm.mp4"))
cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_114505_snow_in2n.mp4"))
texts.append("Snow")

original_cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_113439_farm.mp4"))
cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_114501_sunset_in2n.mp4"))
texts.append("sunset")

original_cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_113439_farm.mp4"))
cams.append(cv2.VideoCapture("renders/farm-small/2023-07-06_114500_storm_in2n.mp4"))
texts.append("storm")


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


for i in range(len(cams)):
    cam, text, original_cam = cams[i], texts[i], original_cams[i]

    text_embedd = get_single_text_embedding(text)


    text_img_avg = 0
    cnt = 0
    consistency_avg = 0
    consistency_in2n_avg = 0
    psnr_avg = 0
    clip_image_sim_avg = 0
    niqe_avg = 0

    last_frame_embedd = None
    last_original_embedd = None
    while(True):
        ret,frame = cam.read()
        _, original_frame = original_cam.read()

        if ret:
            # # downsizing 4 times
            # current_shape = frame.shape
            # factor = 4
            # frame = cv2.resize(frame, dsize=(current_shape[0]//factor ,current_shape[1]//factor), interpolation=cv2.INTER_LINEAR)
            # original_frame = cv2.resize(original_frame, dsize=(current_shape[0]//factor ,current_shape[1]//factor), interpolation=cv2.INTER_LINEAR)

            frame_embedd = get_single_image_embedding(frame)
            original_frame_embedd = get_single_image_embedding(original_frame)
            text_img_avg += cosine_similarity(frame_embedd, text_embedd)
            if last_frame_embedd is not None:
                consistency_avg += cosine_similarity(last_frame_embedd, frame_embedd)
                consistency_in2n_avg += cosine_similarity(last_frame_embedd - last_original_embedd, frame_embedd - original_frame_embedd)
            cnt += 1
            last_frame_embedd = frame_embedd
            last_original_embedd = original_frame_embedd

            cv2.imwrite("test_metric.png", frame)
            niqe_avg += niqe("test_metric.png", None)

            clip_image_sim_avg += cosine_similarity(frame_embedd, original_frame_embedd)
            psnr_avg += PSNR(original_frame, frame)
        else:
            break

    print("___________________________________________")
    print(text)
    print(text_img_avg / cnt, consistency_avg / (cnt - 1), consistency_in2n_avg / (cnt - 1))
    print("CLIP Image Sim:", clip_image_sim_avg / cnt)
    print("PSNR:", psnr_avg / cnt)
    print("NIQE:", niqe_avg / cnt)
    
    cam.release()
