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
        parser.add_argument('--sketch_temp_data', type=str, default="TU-Berlin-sketch/temp-filelist.txt"  ,help='Path to txt file containing few sketch paths for testing.')

        parser.add_argument('--img_path', type=str, default="TU-Berlin-images/" , help='Image-Dataset path.')
        parser.add_argument('--img_ext', type=str, default=".JPEG"  ,help='Image-Dataset Image extention.')
        parser.add_argument('--img_all_data', type=str, default="TU-Berlin-images/filelist.txt"  ,help='Path to txt file containing all image paths.')
        parser.add_argument('--img_temp_data', type=str, default="TU-Berlin-images/temp-filelist.txt"  ,help='Path to txt file containing few image paths for testing.')


        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
