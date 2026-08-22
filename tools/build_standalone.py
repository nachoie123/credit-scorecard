#!/usr/bin/env python3
"""Bake index.html + scorecard_results.json into one self-contained page.

The explainer normally fetches its payload — from the local Python server if
one is listening, from the results file otherwise. Neither is available to a
host that only takes a single HTML file, so this inlines the payload and
short-circuits the loader. It is generated, never hand-edited: regenerate it
whenever the page or the model changes.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "index.html")
DATA = os.path.join(HERE, "scorecard_results.json")


def build() -> str:
    html = open(SRC, encoding="utf-8").read()
    payload = json.load(open(DATA, encoding="utf-8"))

    # The host supplies <!doctype>, <html>, <head> and <body>; this file is
    # dropped inside them, so shipping our own would nest a document.
    head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
    body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)

    # The platform sets its own CSP over the whole frame; a second, stricter
    # policy declared here can only subtract from it.
    head = re.sub(r'<meta http-equiv="Content-Security-Policy".*?>', "", head, flags=re.S)
    head = re.sub(r'<meta charset[^>]*>|<meta name="viewport"[^>]*>', "", head)
    # The gallery lists this beside every other artifact, where the page needs
    # a name rather than a sentence. The subtitle belongs in the listing.
    head = re.sub(r"<title>.*?</title>", "<title>Credit Scorecard</title>",
                  head, flags=re.S)

    # </script> inside the JSON would close the tag it is sitting in.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    chain = re.search(
        r'fetch\("api/analyze"\).*?\.then\(d => \{', body, re.S)
    if not chain:
        sys.exit("could not find the load chain in index.html — did it change?")
    body = body.replace(
        chain.group(0),
        "const SCORECARD_DATA = " + blob + ";\n"
        "/* Baked in: there is no server here and no file beside us. */\n"
        "Promise.resolve(SCORECARD_DATA).then(d => {")

    return head.strip() + "\n" + body.strip() + "\n"


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "standalone.html")
    open(out, "w", encoding="utf-8").write(build())
    print(f"{out}  ({os.path.getsize(out)/1024:.0f} KB)")
