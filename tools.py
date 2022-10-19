import logging
import os
import shutil
import tarfile
import tempfile
from distutils.dir_util import copy_tree

import numpy as np
from git import Repo
from sklearn import metrics


def backup_code(root_dir: str, obj_dir: str):
    """
    Backing up code with git
    :param root_dir: project root directory
    :param obj_dir: object directory which stores backup files
    """
    root_dir = os.path.realpath(root_dir)
    obj_dir = os.path.realpath(obj_dir)
    os.makedirs(obj_dir, exist_ok=True)
    try:
        repo = Repo(root_dir)
    except:
        raise ValueError("root_dir must be directory containing git")
    changed_files = [item.a_path for item in repo.index.diff(None)]
    untracked_files = repo.untracked_files
    files_to_copy = changed_files + untracked_files
    with tempfile.NamedTemporaryFile() as fp:
        repo.archive(fp)
        with tarfile.open(fp.name) as t:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(t, path=obj_dir)
    for changed_file in files_to_copy:
        if os.path.exists(changed_file):
            shutil.copy(changed_file, os.path.join(obj_dir, changed_file))
        else:  # file was deleted in git repo
            os.remove(os.path.join(obj_dir, changed_file))
    copy_tree(os.path.join(root_dir, ".git"), os.path.join(obj_dir, ".git"))


def setup_logger(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def f1_exam(target: np.ndarray, pred: np.ndarray):
    def calc_f1(target_example, pred_example):
        inter = np.logical_and(target_example.astype('uint8'), pred_example.astype('uint8')).sum()
        union = target_example.astype('uint8').sum() + pred_example.astype('uint8').sum()
        if union == 0:
            return 1
        return 2 * inter / union

    num_examples = target.shape[0]
    f1 = []
    for i in range(num_examples):
        temp_f1 = calc_f1(target[i], pred[i])
        f1.append(temp_f1)
    return np.mean(f1)


def NDCG(pred_scores, true_scores, k=8, size_average=False):
    def DCG(y_true, y_score, _k):
        if max(y_true) != 0:
            y_true = np.array(y_true) / max(y_true)
        if max(y_score) != 0:
            y_score = np.array(y_score) / max(y_score)
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:_k])
        gain = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)

    batch_size = pred_scores.shape[0]
    ndcgs = []
    for i in range(batch_size):
        ps = pred_scores[i].tolist()
        ts = true_scores[i].tolist()
        dcg = DCG(ts, ps, k)
        idcg = DCG(ts, ts, k)
        if idcg == 0:
            continue
        ans = dcg / idcg
        ndcgs.append(ans)
    not_nan_idx = np.logical_not(np.isnan(ndcgs))
    ndcgs = np.array(ndcgs)[not_nan_idx]
    if size_average:
        return np.mean(ndcgs)
    return np.sum(ndcgs), len(ndcgs)


def MAP(pred_scores, true_scores, size_average=True):
    def single_map(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true > 0
        if mask.sum() == 0:
            return None
        idx = np.argsort(y_pred)[::-1]
        mask = mask[idx]
        _ret = np.where(mask)[0] + 1
        return ((np.arange(len(_ret)) + 1) / _ret).mean()

    batch_size = pred_scores.shape[0]
    maps = []
    cnt = 0
    for i in range(batch_size):
        ps = pred_scores[i].tolist()
        ts = true_scores[i].tolist()
        ret = single_map(ts, ps)
        if ret is None:
            continue
        maps.append(ret)
        cnt += 1
    if size_average:
        return np.mean(maps)
    return np.sum(maps), cnt


def evaluate_classification(pred: np.ndarray, target: np.ndarray):
    micro_f1_score = metrics.f1_score(target, pred, average='micro')
    # micro_precision = metrics.precision_score(target, pred, average='micro')
    # micro_recall = metrics.recall_score(target, pred, average='micro')

    macro_f1_score = metrics.f1_score(target, pred, average='macro')
    # macro_precision = metrics.precision_score(target, pred, average='macro')
    # macro_recall = metrics.recall_score(target, pred, average='macro')
    exam_f1_score = f1_exam(target, pred)

    hl_loss = metrics.hamming_loss(target, pred)
    return {
        "micro_f1": micro_f1_score,
        # "micro_P": micro_precision,
        # "micro_R": micro_recall,
        "macro_f1": macro_f1_score,
        "exam_f1_score": exam_f1_score,
        # "macro_P": macro_precision,
        # "macro_R": macro_recall,
        "hl_loss": hl_loss,
    }


def evaluate_ranking(pred: np.ndarray, target: np.ndarray, label_ids: np.ndarray):
    ndcg_score = NDCG(pred, target, size_average=True)
    map_score = MAP(pred, target, size_average=True)
    ranking_loss = metrics.label_ranking_loss(label_ids.astype('int64'), pred)
    return {
        "rk_loss": ranking_loss,
        "ndcg": ndcg_score,
        "map": map_score
    }


if __name__ == '__main__':
    pass
