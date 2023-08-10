#!/usr/bin/env python3

import os
import re
import json
import tempfile
import subprocess
from zipfile import ZipFile
from http import client
import yaml

#HOST = 'https://dev-data.spiceai.io'
#API_KEY = '333434|509ac91b5e634695a7f98fe247fe6968' # dev cerexhe spice-app-template

HOST = 'https://data.spiceai.io'
API_KEY = os.environ['SPICE_API_KEY']

def make_conn():
  conn = client.HTTPSConnection if HOST.startswith('https://') else client.HTTPConnection
  return conn(re.sub('https?://', '', HOST))

def get(conn, path):
  conn.request('GET', path, '', {})
  return handle(conn)

def post(conn, path, data):
  print(json.dumps(data, indent=2))
  conn.request('POST', path, json.dumps(data), {
    'Content-type': 'application/json',
  })
  return handle(conn)

def handle(conn):
  rs = conn.getresponse()
  r = rs.read()
  try:
    j = json.loads(r)
    return j
  except:
    return r

def train():
  conn = make_conn()
  tq = '''SELECT number as "ts", CAST(base_fee_per_gas / 1000000000.0 AS DOUBLE) as "y" from eth.recent_blocks
WHERE base_fee_per_gas IS NOT NULL'''
  iq = '''SELECT number as "ts", CAST(base_fee_per_gas / 1000000000.0 AS DOUBLE) as "y" from eth.recent_blocks
WHERE base_fee_per_gas IS NOT NULL ORDER BY ts DESC LIMIT 10'''

  tq += ' ORDER BY number DESC LIMIT 500' # last 500 blocks

  with tempfile.TemporaryDirectory() as tmp:
    zpath = f'{tmp}/function.zip'
    if not os.path.exists(zpath):
      print('creating zip ...')
      with ZipFile(zpath, 'w') as z:
        for filename in ['linear_regression.py', 'requirements.txt']:
          with open(filename, encoding='utf8') as f:
            s = f.read()
            z.writestr(filename, s)

    out = subprocess.run(['ipfs', '--api', '/dns/localhost/tcp/5001', 'add', '-q', zpath], capture_output=True, text=True)
    code_package_cid = out.stdout.strip()
    print(code_package_cid)

  config = {
    "model_type": 'user_linear_regression',
    "lookback_size": 5,
    "forecast_size": 1,
    "epochs": 1,
    "train_query": tq,
    "inference_query": iq,
    'metadata': {
      "firecache": True,
    },
    'runtime': 'python3.10',
    'code_package_cid': code_package_cid,
    'train_handler': 'linear_regression.train',
    'inference_handler': 'linear_regression.infer',
  }
  with open('context.yaml', 'w', encoding='utf8') as f:
    yaml.dump(config, f)

  rs = post(conn, f'/v0.1/train?api_key={API_KEY}', config)
  print(rs)

if __name__ == '__main__':
  train()

