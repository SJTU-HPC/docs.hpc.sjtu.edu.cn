"""URL checking script

This script loops over all static HTML pages generated on the NERSC
documentation website, and validates (i.e., resolves) every URL on every page.
It throws an error if even one URL fails.

A cached list of known good URLs is supported via good-url-cache.txt
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup
import validators
import sys

# Known good pages that do not need to be validated. More are appended to this
# list as the script crawls the docs website so that we do not re-validate the
# same pages.
badlist  = []
goodlist = ["https://www.lbl.gov/disclaimers",
            "https://science.energy.gov",
            "https://www.lbl.gov",
            "https://nxcloud01.nersc.gov",
            "http://epsi.pppl.gov/xgc-users/how-to-become-an-xgc-user",
            "https://stash.nersc.gov",
            "https://stash.nersc.gov:8443",
            "https://www.nersc.gov",
            "http://localhost",
            "https://localhost",
            "http://localhost:5000",
            "https://localhost:5000",
            "https://registry.services.nersc.gov",
            "https://rancher.spin.nersc.gov/v2-beta/projects/1a5/services/NotMyStack"]
skiplist = ["https://doi.org/"]

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
        url_raw, end_quote = get_url(page)
        page = page[end_quote:]
        if url_raw:

            url = url_raw.rstrip("/")
            
            if any(suburl in url for suburl in skiplist):
                print("SKIP: {}".format(url))
                continue

            if not validators.url(url):
                print("INVALID: {}".format(url))
                continue
            
            try:
                if url not in goodlist:
                    requests.get(url, timeout=60)
                    goodlist.append(url)
                    print("OK: {}".format(url))
                    
            except requests.exceptions.ConnectionError as ex:
                print("BAD: ", url)
                print("INFO:", ex)
                badlist.append(url)
        else:
            break


def main():
    """Loops over all documentation pages and checks validity of URLs."""
    parser = argparse.ArgumentParser(description="Validate some URLs.")

    parser.add_argument("doc_base_dir", metavar="doc_base_dir", type=str,
                        help="Base directory of NERSC documentation site.")

    parser.add_argument("--goodurls", type=str,
                        default="good-urls-cache.txt",
                        help="File with list of good urls (to skip)")

    args = parser.parse_args()

    if os.path.isfile(args.goodurls):
        with open(args.goodurls) as f:
            global goodlist
            goodlist += f.read().splitlines()
            print("read cached good urls from {}".format(args.goodurls))
    for url in goodlist:
        print("GOOD: {}".format(url))
    
    print("Checking pages for valid URLs ...")
    doc_root_dir = args.doc_base_dir
    for root, dirs, filenames in os.walk(doc_root_dir):
        for each_file in filenames:
            if each_file.endswith(".html"):
                filepath = root + os.sep + each_file
                print("   ", filepath, "...")
                filehandle = open(filepath, "r")
                mypage = filehandle.read()
                page = str(BeautifulSoup(mypage, "html.parser"))
                check_url(page)

    with open(args.goodurls, "w") as f:
        f.write('\n'.join(goodlist))

    print("SUMMARY:")
    if len(badlist) > 0:
        print("Failed urls:")
        for url in badlist:
            print(url)
        return 1
    else:
        print("No bad urls!")
        return 0

sys.exit(main())
