"""Example on how to use the burbank software.
"""

import pandas as pd
import os
from absl import flags
from absl import app
from data_preprocessing import *
from burbank import *

flags.DEFINE_enum('action', None, ['prepare', 'train', 'predict'],
    help='Prepare xlsx data for training or predict results.')
flags.DEFINE_integer('year', 2024, help='The year of the potato clones to classify.')
flags.DEFINE_string('xlsx', 'xlsx_data/', help='The xlsx dataset directory.')
flags.DEFINE_string('input', 'input/', help='The path containing preped input data.')
flags.DEFINE_string('model', 'model/', help='The path containing the trained model.')
flags.DEFINE_string('output', 'output/', help='The output directory.')
FLAGS = flags.FLAGS

def main(_):
    if not os.path.isdir(FLAGS.input):
        os.mkdir(FLAGS.input)
    
    if not os.path.isdir(FLAGS.output):
        os.mkdir(FLAGS.output)
    
    if FLAGS.action == 'prepare':
        # scrape_website_data(station='KFLO', year=2024, start_month=4, start_day=21, end_month=8, end_day=31)
        # scrape_website_data(station='ONTO', year=2024, start_month=4, start_day=21, end_month=8, end_day=31)
        # scrape_website_data(station='HERO', year=2024, start_month=3, start_day=23, end_month=8, end_day=1)
        # scrape_website_data(station='HERO', year=2024, start_month=4, start_day=21, end_month=8, end_day=31)
        potatoes = process_xlsx_data(FLAGS.xlsx, FLAGS.year)
        potatoes.to_csv(os.path.join(FLAGS.input, "potatoes.csv"))
        gdd = make_gdd_dict()
        gdd.to_csv(os.path.join(FLAGS.input, "gdd.csv"))
        
    elif FLAGS.action == 'train':
        input_df = load_preprocessed_data(FLAGS.input)
        learner(df=input_df, output_path=FLAGS.output,
                region='all', max_na_col=500, impute='y', model='svc')
        
    elif FLAGS.action == 'predict':
        input_df = load_preprocessed_data(FLAGS.input)
        input_df.to_csv(os.path.join(FLAGS.output, "processed_potatoes.csv"))
        predict(preprocessed_input=input_df, model_file_path=os.path.join(FLAGS.model, "svc.pkl"), output_path=FLAGS.output)
        
    
if __name__ == '__main__':
      app.run(main)