
import os, datetime

def analyze_timeline(file):

    stats = os.stat(file)

    created = datetime.datetime.fromtimestamp(stats.st_ctime)
    modified = datetime.datetime.fromtimestamp(stats.st_mtime)

    flags = []

    if modified > created:
        flags.append("file modified after creation")

    return flags
