import numpy as np
from PIL import Image
import cv2
import pandas as pd

from data_tools import ops, augment
import ast
import random

import os
from pathlib import Path
from typing import Dict, List, Tuple

# dataset paths
TRAIN_DATA_PATH = Path('dataset/train_images')
TRAIN_CSV = Path('dataset/train.csv')


class FramePoints:
    CATEGORY = 0
    # based on train dataset  (height, width)
    IMG_HEIGHT = 720
    IMG_WIDTH = 1280

    def __init__(self, vals, v_frame, img_id):
        self.vals = vals
        self.v_frame = v_frame
        self.img_id = img_id.split('-')[0]  # string: img_id-video_frame

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'image_id={self.image_id}, ' \
               f'video_frame={self.video_frame}, ' \
               f'n_points={self.__len__()})'

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, index):
        return self.vals[index]

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        try:
            self.item = self.vals[self._counter]
            self._counter += 1
            return self.parse_points(self.item)
        except Exception as _e:
            raise StopIteration()

    @staticmethod
    def parse_points(data: Dict[str, int]) -> Tuple[int, int, int, int]:
        x = data['x']
        y = data['y']
        width = data['width']    # box width
        height = data['height']  # box height
        return x, y, width, height

    @property
    def video_frame(self) -> int:
        return self.v_frame

    @property
    def image_id(self) -> int:
        return self.img_id

    @property
    def image_name(self) -> str:
        # based on train data folder
        return f"video_{self.image_id}/{self.video_frame}.jpg"

    def load_image(self, file_path: Path = TRAIN_DATA_PATH, draw_points: bool = False) -> np.ndarray:
        # img = cv2.imread(f'{file_path / self.image_name}')
        img = load_image(f'{file_path / self.image_name}')

        if draw_points:
            for p in self.vals:
                x1, y1, width, height = self.parse_points(p)
                x2 = x1 + width
                y2 = y1 + height
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

        return img

    def bbox_nparray(self) -> np.ndarray:
        x = []
        for p in self.vals:
            # x1, y1, x2, y2 = ops.bbox_yolo(self.parse_points(p), (self.IMG_HEIGHT, self.IMG_WIDTH))
            x1, y1, x2, y2 = ops.to_darknet_label_format(self.parse_points(p), (self.IMG_HEIGHT, self.IMG_WIDTH))
            x.append([self.CATEGORY, x1, y1, x2, y2])
        return np.array(x, dtype=np.float64)


def load_image(file_path) -> np.ndarray:
    return np.array(Image.open(file_path))


def load_frame_dataset(csv_data_path: Path = TRAIN_CSV) -> List[FramePoints]:

    df = pd.read_csv(csv_data_path)
    frame_data = [
        FramePoints(ast.literal_eval(ann), v_frame, img_id)
        for ann, v_frame, img_id in zip(df['annotations'], df['video_frame'], df['image_id'])
    ]
    return frame_data


def create_train_data(split_ratio=0.80, augment_data: bool = False):

    # dataset path to be used for training
    base_path = Path('starfish_dataset')
    image_train_dir = base_path / 'images' / 'train'
    image_val_dir = base_path / 'images' / 'val'

    labels_train_dir = base_path / 'labels' / 'train'
    labels_val_dir = base_path / 'labels' / 'val'

    # create paths
    for p in [image_train_dir, image_val_dir, labels_train_dir, labels_val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # load and shuffle data
    frame_data = load_frame_dataset()
    n_images = len(frame_data)
    train_n = int(n_images * split_ratio)
    val_n = n_images - train_n
    print(f'number of images: {n_images}...')
    print(f'number of train images: {train_n}\n'
          f'number of val images:   {val_n}\n')

    # shuffle and split
    random.shuffle(frame_data)
    train_data = frame_data[:train_n]
    val_data = frame_data[train_n:]

    def re_save(frame_list, image_save_path: Path, label_save_path: Path, text_file: str):

        text_file_fp = base_path / text_file
        if os.path.exists(text_file_fp):
            os.remove(text_file_fp)
        os.mknod(text_file_fp)
        n = len(frame_list)
        c = 1
        for frame in frame_list:

            img = frame.load_image()
            labels = frame.bbox_nparray()
            if augment_data and frame:  # augment only non-empty frames

                augments = augment.run_augment(img=img.copy())
                for idx, aug in enumerate(augments):
                    image_a_fn = f'{frame.image_id}_{frame.video_frame}_{idx}.jpg'
                    label_a_fn = f'{frame.image_id}_{frame.video_frame}_{idx}.txt'
                    aug.save(image_save_path / image_a_fn)
                    np.savetxt(label_save_path / label_a_fn, labels, fmt='%g ')

            # file names
            image_fn = f'{frame.image_id}_{frame.video_frame}.jpg'
            label_fn = f'{frame.image_id}_{frame.video_frame}.txt'
            Image.fromarray(img).save(image_save_path / image_fn)
            np.savetxt(label_save_path / label_fn, labels, fmt='%g ')

            # verify the points are valid
            ops.verify_points(label_save_path / label_fn)

            # match coco 'train2017.txt'/'val2017.txt'
            with open(text_file_fp, 'a', encoding='utf-8') as out:
                out.write(f'{str(image_save_path / image_fn)}\n')
            print(f'image/labels updated: {c}/{n}....', end='\r')
            c += 1

    # create dataset
    re_save(train_data, image_train_dir, labels_train_dir, 'train.txt')
    re_save(val_data, image_val_dir, labels_val_dir, 'val.txt')


def main():
    create_train_data(augment_data=True)


if __name__ == "__main__":
    main()
