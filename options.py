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

        parser.add_argument('--epochs', type=int, default=5 ,help='Training epochs.')
        parser.add_argument('--logdir', type=str, default="Log/" ,help='Directory to log in exps for TensorBoard')

        parser.add_argument('--hashcode_length', type=int, default=64 ,help='Length of the final output hash code')
        parser.add_argument('--sketch_model', type=str, default="saved_models/sketch/"  ,help='Path to the saved sketch model')
        parser.add_argument('--image_model', type=str, default="saved_models/image/"  ,help='Path to saved image model.')


        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
