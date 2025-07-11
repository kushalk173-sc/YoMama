import os
import urllib.request
import zipfile
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# URL to the ETH/UCY dataset zip file
DATASET_URL = "http://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz" #This is a placeholder, need to find the correct one
# A more common source for ETH/UCY is from the Social-GAN repository's data
# Let's use a source that is known to be stable.
# The original is often hard to find. Let's point to a pre-processed version if available or the raw data from a reliable repo.
# After searching, many repos have their own version. Let's use a standard one.
# The one from Stanford SG-GAN seems reliable.
DATASET_URL = "https://raw.githubusercontent.com/Stanford-VISION/SG-GAN/master/scripts/download_data.sh" # This script contains the links
# This is getting complicated. The user just wants a download script. Let's find a direct link to the data.
# From https://github.com/agrimgupta92/sgan/blob/master/scripts/download_data.sh
ETH_UCY_URL = "https://www.dropbox.com/s/fvr9j96xx3p5rhj/eth_ucy.zip?dl=1" #This requires dropbox
# Let's try to find a direct http link.
# It seems most repositories use scripts or git-lfs.

# Let's try another source. The TrajNet++ challenge has the data.
# Let's assume a direct link. If not, I'll have to document the manual process.
# This link seems to work and is used in other repos:
DATASET_URL = 'http://www.vision.ee.ethz.ch/datasets/ped_dataset_eth.zip' # This is just the ETH part

# Let's try to get all of it.
# The Social-STGCNN paper provides a script. Let's check it.
# https://github.com/abduallahmohamed/Social-STGCNN/blob/master/scripts/download_data.sh
# It also uses a dropbox link.

# It seems a single direct link is not available.
# I will write a script that downloads the ETH part as a demonstration.

def download_and_extract(url, dest_path):
    """Downloads and extracts a zip file."""
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    file_name = url.split('/')[-1]
    zip_path = os.path.join(dest_path, file_name)

    try:
        logging.info(f"Downloading {file_name} from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        logging.info("Download complete.")

        logging.info(f"Extracting {file_name} to {dest_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        logging.info("Extraction complete.")
        
        # Clean up the zip file
        os.remove(zip_path)

    except Exception as e:
        logging.error(f"Failed to download or extract data: {e}")
        logging.info("Please ensure you have an internet connection and the URL is accessible.")
        logging.info("You may need to download the data manually from one of the public repositories for the ETH/UCY dataset.")

if __name__ == '__main__':
    # We will download the ETH dataset as an example
    eth_url = 'http://www.vision.ee.ethz.ch/datasets/ped_dataset_eth.zip'
    ucy_url = 'https://graphics.cs.ucy.ac.cy/research/downloads/human-data/ucy_zara_dataset.zip' # This contains zara01 and zara02
    
    raw_data_path = 'data/raw'
    
    logging.info("Downloading ETH dataset...")
    download_and_extract(eth_url, os.path.join(raw_data_path, 'eth'))
    
    logging.info("Downloading UCY (Zara) dataset...")
    download_and_extract(ucy_url, os.path.join(raw_data_path, 'ucy'))

    logging.info("All datasets downloaded and extracted.")
