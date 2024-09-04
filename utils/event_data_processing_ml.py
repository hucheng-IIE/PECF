import numpy as np
import random
import pandas as pd
import os

def generate_data_ml(inPath, fileName, fileName2, fileName3, target_rels, seq_len, rel_types):
  last_time = -1
  x_day = [0 for _ in range(rel_types)]
  has_Y = 0
  y_data = []
  x_data = []

  with open(os.path.join(inPath, fileName), 'r') as fr:
      for line in fr:
          line_split = line.split()
          rel = int(line_split[1])
          time = int(line_split[3])

          if time > last_time:
            for blank in range(last_time+1, time):
              y_data.append(0)
              x_data.append([])
            if last_time > -1:
              y_data.append(has_Y)
              x_data.append(x_day)
            last_time = time
            has_Y = 0
            x_day = [0 for _ in range(rel_types)]
          if rel in target_rels:
            has_Y = 1
          x_day[rel] += 1

  with open(os.path.join(inPath, fileName2), 'r') as fr:
      for line in fr:
          line_split = line.split()
          rel = int(line_split[1])
          time = int(line_split[3])

          if time > last_time:
            for blank in range(last_time+1, time):
              y_data.append(0)
              x_data.append([])
            if last_time > -1:
              y_data.append(has_Y)
              x_data.append(x_day)
            last_time = time
            has_Y = 0
            x_day = [0 for _ in range(rel_types)]
          if rel in target_rels:
            has_Y = 1
          x_day[rel] += 1

  with open(os.path.join(inPath, fileName3), 'r') as fr:
      for line in fr:
          line_split = line.split()
          rel = int(line_split[1])
          time = int(line_split[3])

          if time > last_time:
            for blank in range(last_time+1, time):
              y_data.append(0)
              x_data.append([])
            if last_time > -1:
              y_data.append(has_Y)
              x_data.append(x_day)
            last_time = time
            has_Y = 0
            x_day = [0 for _ in range(rel_types)]
          if rel in target_rels:
            has_Y = 1
          x_day[rel] += 1
    
  y_data.append(has_Y)
  x_data.append(x_day)
  #time->type count 2584 time
  x_cum_data = []
  for time in range(len(y_data)):
      x_cum_day = [0 for _ in range(rel_types)]
      if time < seq_len:
        for before_time in range(time+1):
           x_cum_day = [i+j for i,j in zip(x_cum_day,x_data[before_time])]
      else:
        for before_time in range(time-seq_len,time+1):
           x_cum_day = [i+j for i,j in zip(x_cum_day,x_data[before_time])]
      x_cum_data.append(x_cum_day)

  return x_cum_data, y_data