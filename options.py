"""
    Parse input arguments
"""

import argparse

class Options():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Zero-Shot Image Hashing model',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--sketch_path', type=str, default="TU-Berlin-sketch/" , help='Sketch-Dataset path.')
        parser.add_argument('--sketch_ext', type=str, default=".png"  ,help='Sketch-Dataset Image extention.')
        parser.add_argument('--sketch_all_data', type=str, default="TU-Berlin-sketch/filelist.txt"  ,help='Path to txt file containing all sketch paths.')
        parser.add_argument('--sketch_train_data', type=str, default="TU-Berlin-sketch/filelist-train.txt"  ,help='Path to txt file containing all train-sketch paths.')
        parser.add_argument('--sketch_valid_data', type=str, default="TU-Berlin-sketch/filelist-valid.txt"  ,help='Path to txt file containing all valid-sketch paths.')
        parser.add_argument('--sketch_test_data', type=str, default="TU-Berlin-sketch/filelist-test.txt"  ,help='Path to txt file containing all test-sketch paths.')

        parser.add_argument('--img_path', type=str, default="TU-Berlin-images/" , help='Image-Dataset path.')
        parser.add_argument('--img_ext', type=str, default=".JPEG"  ,help='Image-Dataset Image extention.')
        parser.add_argument('--img_all_data', type=str, default="TU-Berlin-images/filelist.txt"  ,help='Path to txt file containing all image paths.')
        parser.add_argument('--img_train_data', type=str, default="TU-Berlin-images/filelist-train.txt" , help='Path to txt file containing all train-image paths.')
        parser.add_argument('--img_valid_data', type=str, default="TU-Berlin-images/filelist-valid.txt" , help='Path to txt file containing all valid-image paths.')
        parser.add_argument('--img_test_data', type=str, default="TU-Berlin-images/filelist-test.txt" , help='Path to txt file containing all test-image paths.')


        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
