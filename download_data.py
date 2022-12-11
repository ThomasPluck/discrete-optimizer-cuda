from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import urllib

try:
    from urllib.error import URLError
    from urllib.request import urlretrieve
except ImportError:
    from urllib3 import URLError
    from urllib import urlretrieve

valid_datasets = {
    "mnist": [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ],
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
}


def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


def download(destination_path, url, quiet):
    if os.path.exists(destination_path):
        if not quiet:
            print("{} already exists, skipping ...".format(destination_path))
    else:
        print("Downloading {} ...".format(url))
        try:
            hook = None if quiet else report_download_progress
            urlretrieve(url, destination_path, reporthook=hook)
        except URLError:
            raise RuntimeError("Error downloading resource!")
        finally:
            if not quiet:
                # Just a newline.
                print()


def unzip(zipped_path, quiet):
    unzipped_path = os.path.splitext(zipped_path)[0]
    if os.path.exists(unzipped_path):
        if not quiet:
            print("{} already exists, skipping ... ".format(unzipped_path))
        return
    with gzip.open(zipped_path, "rb") as zipped_file:
        with open(unzipped_path, "wb") as unzipped_file:
            unzipped_file.write(zipped_file.read())
            if not quiet:
                print("Unzipped {} ...".format(zipped_path))


def main():
    parser = argparse.ArgumentParser(description="Download datasets from the internet")
    parser.add_argument(
        "-d",
        "--dataset",
        default="",
        help="Desired dataset",
        choices=valid_datasets.keys(),
    )
    parser.add_argument(
        "-o", "--output", default="./.data", help="Destination directory"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Don't report about progress"
    )
    options = parser.parse_args()

    if options.dataset not in valid_datasets:
        raise NotImplementedError("{} is not a valid dataset".format(options.dataset))

    if not os.path.exists(options.output + "/" + options.dataset):
        os.makedirs(options.output + "/" + options.dataset)

    try:
        for resource in valid_datasets[options.dataset]:
            path = os.path.join(
                options.output + "/" + options.dataset, resource.split("/")[-1]
            )
            url = resource
            download(path, url, options.quiet)
            unzip(path, options.quiet)
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()
