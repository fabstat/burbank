"""Example on how to use the burbank software.
"""

import pandas as pd
import os
from absl import flags
from absl import app
from data_preprocessing import *
from burbank import *

flags.DEFINE_enum('action', None, ['prepare', 'predict'],
    help='Prepare xlsx data for training or predict results.')
flags.DEFINE_string('xlsx', 'xlsx_data/', help='The xlsx dataset directory.')
flags.DEFINE_string('gdd', 'gdd/', help='The path containing GDD data.')
flags.DEFINE_string('input', 'input/', help='The path containing preped input data.')
flags.DEFINE_string('output', 'output/', help='The output directory.')
FLAGS = flags.FLAGS

def main(_):
    if not os.path.isdir(FLAGS.input):
        os.mkdir(FLAGS.input)
    
    if not os.path.isdir(FLAGS.output):
        os.mkdir(FLAGS.output)
    
    if FLAGS.action == 'prepare':
        potatoes = process_xlsx_data(FLAGS.xlsx)
        potatoes.to_csv(os.path.join(FLAGS.input, "potatoes.csv"))
        gdd = make_gdd_dict(FLAGS.gdd)
        gdd.to_csv(os.path.join(FLAGS.input, "gdd.csv"))
        
    elif FLAGS.action == 'predict':
        input_df = load_preprocessed_data(FLAGS.input)
        learner(df=input_df, output_path=FLAGS.output, new_data=None, 
                region='all', max_na_col=50, impute='y', model='svc')
        
    
if __name__ == '__main__':
      app.run(main)