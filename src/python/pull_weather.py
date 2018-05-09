import sys
import os
import urllib3
import json
import numpy as np
import datetime
import time
import certifi
import random

reqfile = sys.argv[1]
grid_size = sys.argv[2]

basepath = "/home/chase/projects/peakload/data/weather/ercot/"
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

log_file = open(basepath + "requests/" + str(random.randint(1000,9999)) + ".log", 'w')

prog = 0
with open(reqfile, 'r') as d:
    lines = d.readlines()
    for line in lines:
        cureq = line.strip()
        tokens = cureq.split("/")
        tokens = tokens[-1].split(",")
        lat = tokens[1]
        lon = tokens[0]
        posix = int(tokens[2].split("?")[0])
        year = str(datetime.datetime.fromtimestamp(posix).year)
        #outpath = basepath + "/grid" + str(grid_size) + "/" + year + "/" + str(posix)
        outpath = basepath + "major_cities/" + year + "/" + str(posix)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        try:
            response = http.request('GET', cureq)
            #with open(outpath + "/grid_" + str(lat) + "," + str(lon) + ".json", 'w') as f:
            with open(outpath + "/city_" + str(lat) + "," + str(lon) + ".json", 'w') as f:
                payload = json.loads(response.data)
                out = json.dumps(payload)
                f.write(out)
        except Exception as err:
            log_file.write(str(err))
            log_file.write(cureq + "\n")
        prog += 1
        perc = (prog/len(lines))*100
        if perc % 1 == 0:
            print(str(perc) + "% downloaded")
            
log_file.close()
