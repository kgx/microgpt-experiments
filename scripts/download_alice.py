"""
Download Alice's Adventures in Wonderland (Project Gutenberg) to data/alice/alice.txt.
Run from project root: python scripts/download_alice.py
"""

import os
import urllib.request

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "data", "alice", "alice.txt")
URL = "https://www.gutenberg.org/files/11/11-0.txt"

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, OUTPUT)
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    main()
