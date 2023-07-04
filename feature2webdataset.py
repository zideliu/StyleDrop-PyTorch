import json
import os
import sys
import shutil
import numpy as np
import datetime
import webdataset as wds
from multiprocessing import Process
from PIL import Image

def make_wds_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))

    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    url = '/'.join(fname.split('/')[-2:])
    sink = wds.TarWriter(fname, **kwargs)
    for item in samples:
        for content in map_func(item, url):
            sink.write(content)
    sink.close()

if __name__ == "__main__":
    
    # all files in img_emb_path and text_raw_path are npy format
    # files in img_emb_path and text_raw_path are related by their file names
    # i.e., img_emb/000000.npy and text_raw/000000.npy are related, img_emb/000001.npy and text_raw/000001.npy are related
    img_emb_path = /path/to/img_emb
    text_raw_path = /path/to/text_raw
    
    num_workers = 180
    img_filelist = os.listdir(img_emb_path)
    text_raw_filelist = os.listdir(text_raw_path)

    img_file_paths = [os.path.join(img_emb_path, fp) for fp in img_filelist]
    text_raw_file_paths = [os.path.join(text_raw_path, fp) for fp in text_raw_filelist]

    text_raw_file_paths = sorted(text_raw_file_paths)
    img_file_paths = sorted(img_file_paths)

    print("Num of img_file_paths: ", len(img_file_paths))
    print("Num of text_raw_file_paths: ", len(text_raw_file_paths))

    file_paths = []
    for fi, ftr in zip(img_file_paths, text_raw_file_paths):
        file_paths.append([fi, ftr])

    num_shards = len(file_paths)

    print(file_paths)

    def sampler(fp, url):
        image_path, text_raw_path = fp
        text_raw = np.load(text_raw_path, allow_pickle=True)
        images = np.load(image_path, allow_pickle=True).reshape(-1, 256)

        print(f"shape of input raw text: {text_raw.shape} with dtype: {text_raw.dtype} and shape of images: {images.shape} with dtype: {images.dtype}")

        for i, (img_emb, text) in enumerate(zip(images, text_raw)):
            try:
                
                sample = {
                    "__key__": f"%08d"%i,
                    "__url__": url, # path/to/xxx.tar
                    "image.npy": img_emb.tobytes(),
                    "text.npy": str(text),
                }

                yield sample
            except:
                continue

    make_wds_shards(
        pattern=f"{output_path}/%08d.tar",
        num_shards=num_shards,
        num_workers=num_workers,
        samples=file_paths,
        map_func=sampler,
    )

