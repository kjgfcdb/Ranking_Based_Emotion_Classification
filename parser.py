import json
import os
import re
from xml.etree.ElementTree import ElementTree
from os.path import join

import pkuseg
from tqdm import tqdm

EMOTIONS = ("Joy", "Hate", "Love", "Sorrow",
            "Anxiety", "Surprise", "Anger", "Expect")


def Ren_CECps_parser(dataset_path: str, output_path: str, stopwords_path: str):
    """convert Ren_CECps data to json format"""
    json_file = open(output_path, "w")
    e2i = {e: i for i, e in enumerate(EMOTIONS)}
    seg = pkuseg.pkuseg(model_name="web")
    with open(stopwords_path) as f:
        stopwords = [word.strip() for word in f if word.strip()]

    def _print(line):
        json_file.write(line + "\n")

    for xml_file in tqdm(os.listdir(dataset_path), desc="Parsing xml files of Ren_CECps dataset", leave=True):
        xml_file = join(dataset_path, xml_file)
        root = ElementTree(file=xml_file).getroot()
        for elem in root.iter(tag="sentence"):
            children = (child for child in elem)
            emotions = list(filter(lambda child: child.tag in EMOTIONS, children))
            sentence = elem.attrib['S']
            emotions_list = list(range(len(EMOTIONS)))
            for emotion in emotions:
                emotions_list[e2i[emotion.tag]] = float(emotion.text)
            # sentence = seg.cut(sentence)
            # # remove stopwords
            # buf = []
            # for word in sentence:
            #     if word not in stopwords:
            #         buf.append(word)
            # sentence = buf

            if len(sentence) <= 4 or len(sentence) > 100:
                continue
            obj = dict(
                text=sentence,
                emotion=emotions_list
            )
            _print(json.dumps(obj))
    json_file.close()


def Sem_Eval_Parser(dataset_path: str, output_path: str):
    json_file = open(output_path, "w")
    sem_eval_emotions = ("anger", "disgust", "fear", "joy", "sadness", "surprise")

    def _print(line):
        json_file.write(line + "\n")

    for folder in os.listdir(dataset_path):
        folder = join(dataset_path, folder)
        if not os.path.isdir(folder):
            continue
        text_file = [f for f in os.listdir(folder) if f.endswith("xml")][0]
        emotion_file = [f for f in os.listdir(folder) if "emotion" in f][0]
        with open(join(folder, text_file)) as f:
            instances = f.readlines()[1:-1]
        with open(join(folder, emotion_file)) as f:
            emotions = f.readlines()
        assert len(emotions) == len(instances)
        for instance, emotion in zip(instances, emotions):
            text = re.findall(r"<.*?>(.*)<.*?>", instance)[0]
            emotion = emotion.strip().split(" ")[1:]
            emotion = {sem_eval_emotions[i]: int(item) for i, item in enumerate(emotion)}
            obj = dict(
                text=text,
                **emotion
            )
            _print(json.dumps(obj))
    json_file.close()


if __name__ == "__main__":
    # Sem_Eval_Parser("data/Semeval.2007", "data/Semeval.2007.json")
    Ren_CECps_parser("data/Ren_CECps", "data/Ren_CECps.json", "data/stopwords.txt")
