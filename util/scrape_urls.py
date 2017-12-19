"""URL checking script

This script loops over all static HTML pages generated on the NERSC
documentation website, and validates (i.e., resolves) every URL on every page.
It throws an error if even one URL fails.
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup
import validators


def get_url(this_page):
    """Print out the URL

    Found on StackOverflow: https://stackoverflow.com/a/15517610

    :param this_page: html of web page
    :return: urls in that page
    """

    # Validate only external URLs, not internal ones. (mkdocs can validate
    # internal links itself.) External URLs have the "http" prefix, whereas
    # internal links user relative paths.
    start_link = this_page.find('a href="http')
    if start_link == -1:
        return None, 0
    start_quote = this_page.find('"', start_link)
    end_quote = this_page.find('"', start_quote + 1)
    this_url = this_page[start_quote + 1: end_quote]
    return this_url, end_quote


def check_url(page):
    """Function that checks the validity of a URL."""
    while True:
        url, end_quote = get_url(page)
        page = page[end_quote:]
        if url:
            if not validators.url(url):
                print("ERROR: INVALID URL")
            try:
                requests.get(url)
            except requests.exceptions.ConnectionError:
                print("Bad URL: ", url)
                raise
        else:
            break
    print("OK")


def main():
    """Loops over all documentation pages and checks validity of URLs."""
    parser = argparse.ArgumentParser(description="Validate some URLs.")

    parser.add_argument("doc_base_dir", metavar="doc_base_dir", type=str,
                        help="Base directory of NERSC documentation site.")

    args = parser.parse_args()

    print("Checking pages for valid URLs ...")
    doc_root_dir = args.doc_base_dir
    for root, dirs, filenames in os.walk(doc_root_dir):
        for each_file in filenames:
            if each_file.endswith(".html"):
                filepath = root + os.sep + each_file
                print("   ", filepath, "...", end=' ', flush=True)
                filehandle = open(filepath, "r")
                mypage = filehandle.read()
                page = str(BeautifulSoup(mypage, "html.parser"))
                check_url(page)

main()
