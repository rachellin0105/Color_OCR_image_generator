# coding=utf-8
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
from tqdm import tqdm
    
'''
Check whether it wasn't written the word on the image because of the font 
and then create the txt file to list the font with the words not matching 
'''

def not_all_white_pixels(image):
    '''
        Returns True if all white pixels or False if not all white
        check if pixel are all white
    '''
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    pixels = cv2.countNonZero(thresh)
    return True if pixels == (H * W) else False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dic',type=str,default='chars.txt',help="input dictionary txt")
    parser.add_argument('--font_dir',type=str,default='./fonts/chinse_jian',help="dir of font")
    parser.add_argument('--output_txt',type=str,default='wrong_dic.txt',help="output txt of wrong words")
    arg = parser.parse_args()

    # key: font ; value: list of word
    wrong_dic = {}
    font_list = os.listdir(arg.font_dir)

    with open( arg.input_dic,'r',encoding="utf8" ) as f:
        words = f.readlines()
        for font in tqdm(font_list):
            for word in words:
                word = word.replace('\n','')
                font_path = os.path.join(arg.font_dir,font)

                # white background
                image = Image.new("RGB",(50,50),"white")
                draw = ImageDraw.Draw(im=image)
                draw.text(xy=(0, 0), text= word, fill=(0,0,0), font=ImageFont.truetype(font_path, 20))

                image = np.array(image) 
                image = image[:, :, ::-1].copy() 

                if not_all_white_pixels(image):
                    if font in wrong_dic :
                        wrong_dic[font].append(word)
                    else:
                        wrong_dic[font] = [word]

    print("Finished checking")
    if not wrong_dic:
        print("All words match with font")
    else:
        with open( arg.output_txt,'w',encoding="utf8" ) as f:
            for font,words in wrong_dic.items():           
                f.write(key+'\t'+' '.join(value)+'\n')

        print("Finished generating wrong words txt")
