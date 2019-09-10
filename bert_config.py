import numpy as np


class Config:
    gradient_accumulation_steps = 1
    train_batch_size = 128
    num_train_epochs = 10
    learning_rate = 3e-5
    warmup_proportion = 0.1
    max_seq_length = 128
    dropout_prob = 0.2
    hidden_size = 768
    output_dir = "models"

    def process_emotion(self, emotion):
        emotion_score = (np.array(emotion)).tolist()
        emotion_label = (np.array(emotion) > 0).astype('float32').tolist()
        return emotion_score, emotion_label


class Ren_CECps_Config(Config):
    dataset_name = "ren_cecps"
    num_labels = 8
    data_dir = "data/Ren_CECps.json"
    pretrained_model_name = "bert-base-chinese"


class Sem_Eval_Config(Config):
    dataset_name = "sem_eval"
    num_labels = 6
    data_dir = "data/Semeval.2007.json"
    pretrained_model_name = "bert-base-uncased"

    def process_emotion(self, emotion):
        emotion_score = (np.array(emotion) / 100.0).tolist()
        emotion_label = (np.array(emotion) > 0).astype('float32').tolist()
        return emotion_score, emotion_label


class Sina_News_Config(Config):
    dataset_name = "sina_news"
    num_labels = 6  # ['moved', 'shocked', 'funny', 'sad', 'novel', 'angry']
    data_dir = "data/sina.json"
    pretrained_model_name = "bert-base-chinese"
    max_seq_length = 256

    def process_emotion(self, emotion):
        total = sum(emotion)
        emotion_score = (np.array(emotion)).tolist()
        if total != 0:
            emotion_label = (np.array(emotion) > 0.1 * total).astype('float32').tolist()
        else:
            emotion_label = (np.array(emotion) > 0).astype('float32').tolist()
        return emotion_score, emotion_label
