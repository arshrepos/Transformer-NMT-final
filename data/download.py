"""Download and extract Multi30k dataset."""

import os
import urllib.request
import gzip
import shutil

URLS = {
    "train.en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz",
    "train.de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz",
    "val.en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz",
    "val.de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.de.gz",
    "test.en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz",
    "test.de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.de.gz",
}


def download_multi30k(data_dir="data/multi30k"):
    os.makedirs(data_dir, exist_ok=True)

    for filename, url in URLS.items():
        out_path = os.path.join(data_dir, filename)
        if os.path.exists(out_path):
            print(f"  Already exists: {out_path}")
            continue

        gz_path = out_path + ".gz"
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, gz_path)

        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"  Saved: {out_path}")

    print("Multi30k download complete.")


if __name__ == "__main__":
    download_multi30k()
