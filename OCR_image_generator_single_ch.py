# -*- coding: utf-8 -*-
""" 
 clone form
 @author: zcswdt
 @email: jhsignal@126.com
 @file: Color_OCR_image_generator.py
 @time: 2020/06/24
"""
import cv2
import numpy as np
import time

import random
from PIL import Image,ImageDraw,ImageFont
import os
from sklearn.cluster import KMeans


import argparse

from tools.config import load_config
from noiser import Noiser
from tools.utils import apply
from math import ceil
import pickle
from tqdm import tqdm
import time
from fontfactory import FontFactory

from data_aug import apply_blur_on_output
from data_aug import apply_prydown
from data_aug import apply_lr_motion
from data_aug import apply_up_motion

from warnings import filterwarnings
filterwarnings('ignore')


class FontColor(object):
	def __init__(self, col_file):
		with open(col_file, 'rb') as f:
			u = pickle._Unpickler(f)
			u.encoding = 'latin1'
			self.colorsRGB = u.load()
		self.ncol = self.colorsRGB.shape[0]

		# convert color-means from RGB to LAB for better nearest neighbour
		# computations:
		self.colorsRGB = np.r_[self.colorsRGB[:, 0:3], self.colorsRGB[:, 6:9]].astype('uint8')
		self.colorsLAB = np.squeeze(cv2.cvtColor(self.colorsRGB[None, :, :], cv2.COLOR_RGB2Lab))

def word_in_font(word,unsupport_chars,font_path):
	#print('1',word)
	#sprint('2',unsupport_chars)
	for c in word:
		#print('c',c)
		if c in unsupport_chars:
			print('Retry pick_font(), \'%s\' contains chars \'%s\' not supported by font %s' % (word, c, font_path))  
			return True
		else:
			continue

# 分析图片，获取最适宜的字体颜色
def get_bestcolor(color_lib, crop_lab):
	if crop_lab.size > 4800:
		crop_lab = cv2.resize(crop_lab,(100,16))  #将图像转成100*16大小的图片
	labs = np.reshape(np.asarray(crop_lab), (-1, 3))         #len(labs)长度为160   
	clf = KMeans(n_clusters=8)
	clf.fit(labs)
	
	#clf.labels_是每个聚类中心的数据（假设有八个类，则每个数据标签属于每个类的数据格式就是从0-8），clf.cluster_centers_是每个聚类中心   
	total = [0] * 8
   
	for i in clf.labels_:
		total[i] = total[i] + 1            #计算每个类中总共有多少个数据
 
	clus_result = [[i, j] for i, j in zip(clf.cluster_centers_, total)]  #聚类中心，是一个长度为8的数组
	clus_result.sort(key=lambda x: x[1], reverse=True)    #八个类似这样的数组，第一个数组表示类中心，第二个数字表示属于该类中心的一共有多少数据[[array([242.55732946, 128.1509434 , 122.29608128]), 689], [array([245.03461538, 128.59230769, 125.88846154]), 260],，，，]
  
	color_sample = random.sample(range(color_lib.colorsLAB.shape[0]), 500)   # 范围是（0,9882），随机从这些数字里面选取500个

	
	def caculate_distance(color_lab, clus_result):
		weight = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
		d = 0
		for c, w in zip(clus_result, weight):

			#计算八个聚类中心和当前所选取颜色距离的标准差之和，每个随机选取的颜色当前聚类中心的差值
			d = d + np.linalg.norm(c[0] - color_lab)           
		return d
 
	color_dis = list(map(lambda x: [caculate_distance(color_lib.colorsLAB[x], clus_result), x], color_sample))   #将color_sample中的每个参数当成x传入函数内,color_lib.colorsLAB[x]是一个元组(r,g,b)也就是字体库里面的颜色
	#color_dis 是一个长度为500的列表[[x,y],[],,,,,]，其中[x,y]其中x表示背景色和当前颜色的距离，y表示该颜色的色号  
	color_dis.sort(key=lambda x: x[0], reverse=True)
	color_num = color_dis[0:200]
	color_l = random.choice(color_num)[1]
	#print('color_dis',color_l)
	#color_num=random.choice(color_dis[0:300])
	#print('color_dis[0][1]',color_dis[0][1])
	return tuple(color_lib.colorsRGB[color_l])
	#return tuple(color_lib.colorsRGB[color_dis[0][1]])

def apply_perspective_transform(img,position,chars,best_color,font,output_image_size=64,mask_size=80,max_degree=10):
	'''
	input:
		img: the original image
		max_degree : the max degree of distortion
	return the image apply with perspective_transform
	'''
	# position, type: turple
	x1 = position[0]
	y1 = position[1]

	# draw text on mask
	mask = Image.new('RGBA', (mask_size,mask_size), (0,0,0,0)) 
	draw = ImageDraw.Draw(mask)
	draw.text((5, 5), chars, best_color, font=font)

	# apply perspective transform on mask
	mask = np.array(mask)

	pts1 = np.float32([[0+random.randint(0,max_degree),0+random.randint(0,max_degree)],
	[mask_size-random.randint(0,max_degree),random.randint(0,max_degree)],
	[random.randint(0,max_degree),mask_size-random.randint(0,max_degree)],
	[mask_size-random.randint(0,max_degree),mask_size-random.randint(0,max_degree)]])

	pts2 = np.float32([[0,0],[output_image_size,0],[0,output_image_size],[output_image_size,output_image_size]])
	
	M = cv2.getPerspectiveTransform(pts1,pts2)

	mask = cv2.warpPerspective(mask,M,(output_image_size,output_image_size))

	mask = Image.fromarray(mask)

	#paste mask on bg and do final_crop 
	img.paste(mask,(x1,y1),mask)
	img = img.crop((x1,y1,x1+output_image_size,y1+output_image_size))

	return img

def generate_single_word(img_path,color_lib,font_path,font_unsupport_chars,chars,cf,output_image_size=64,mask_size=80,temp_word_size=70):


	retry = 0
	img = Image.open(img_path)
	if img.mode != 'RGB':
		img = img.convert('RGB')
	w, h = img.size
	
	x1 = random.randint(0, w - mask_size)
	y1 = random.randint(0, h - mask_size)


	#获得字体，及其大小
	font = ImageFont.truetype(font_path, temp_word_size)         

	#不支持的字体文字，按照字体路径在该字典里索引即可        
	unsupport_chars = font_unsupport_chars[font_path] 

	crop_img = img.crop((x1, y1, x1+output_image_size, y1+output_image_size))
	crop_lab = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_RGB2Lab)

	all_in_fonts=word_in_font(chars,unsupport_chars,font_path)
	if (np.linalg.norm(np.reshape(np.asarray(crop_lab),(-1,3)).std(axis=0))>55 or all_in_fonts) and retry<30:  # 颜色标准差阈值，颜色太丰富就不要了
		retry = retry + 1
	if not cf.customize_color:
		best_color = get_bestcolor(color_lib, crop_lab)
	else:
		r = random.choice([7,9,11,14,13,15,17,20,22,50,100])
		g = random.choice([8,10,12,14,21,22,24,23,50,100])
		b = random.choice([6,8,9,10,11,30,21,34,56,100])
		best_color = (r,g,b)   


	img = apply_perspective_transform(img,(x1,y1),chars,best_color,font)

	return img,chars 

def init_wrong_dict(word_false_txt_path):


	with open(word_false_txt_path,'r',encoding='utf-8') as f:
		lines = f.readlines()

	wrong_dict = {}

	for line in lines:
		line = line.replace('\n','')
		line = line.replace('.TTF','')
		line = line.replace('.ttf','').strip()
		[font,words] = line.split('\t')
		words = words.split(' ')
		wrong_dict[font] = words

	return wrong_dict

def get_word_list(word_txt_path):

	if not os.path.exists(word_txt_path):
		assert False, "Chars file not exists."

	voc_list = []
	with open(word_txt_path,'r',encoding="utf8") as f:
		lines = f.readlines()
		voc_list = [line.replace('\n','')for line in lines]
	return voc_list

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	# base argument
	parser.add_argument('--output_dir', type=str, default='output', help='path of output directory ')

	parser.add_argument('--num_per_word', type=int, default=1, help="Number of images per word ")

	parser.add_argument('--output_image_size', type=int, default=64, help="size of output image ")

	parser.add_argument('--bg_path', type=str, default='./background',
						help='The generated text pictures will use the pictures of this folder as the background.')
						
	parser.add_argument('--fonts_path',type=str, default='./fonts/chinse_jian',
						help='The font used to generate the picture')

	parser.add_argument('--chars_file',  type=str, default='chars.txt',
						help='Chars allowed to be appear in generated images')

	parser.add_argument('--word_false_txt_path',  type=str, default='./wrong_dic.txt',
						help='txt file generated from check_font_validate.py')

	# argument for choice best color
	parser.add_argument('--customize_color', action='store_true', help='Support font custom color')

	parser.add_argument('--color_path', type=str, default='./models/colors_new.cp', 
					help='Color font library used to generate text')
	
	# data augmentation
	parser.add_argument('--blur', action='store_true', default=False,
						help="Apply gauss blur to the generated image")

	parser.add_argument('--noise', action='store_true', default=False,
						help="Apply gauss blur to the generated image")   					    
	
	parser.add_argument('--prydown', action='store_true', default=False,
					help="Blurred image, simulating the effect of enlargement of small pictures")

	parser.add_argument('--lr_motion', action='store_true', default=False,
					help="Apply left and right motion blur")
					
	parser.add_argument('--ud_motion', action='store_true', default=False,
					help="Apply up and down motion blur")                        

	parser.add_argument('--config_file', type=str, default='noise.yaml',
					help='Set the parameters when rendering images')
	
	
	cf = parser.parse_args()

	# load chars file 
	ch_list = get_word_list(cf.chars_file)
	chars_str = ''.join(ch_list)

	FF = FontFactory(chars_str,cf.fonts_path)
	fonts_list = FF.get_fonts_list()
	font_unsupport_chars = FF.get_unsupported_chars()

	# Create directory for generated image
	output_dir = os.path.join('.',cf.output_dir)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	#sey up  noise config file
	if cf.noise:
		flag = load_config(cf.config_file) 	
		noiser = Noiser(flag) 
	
	# 读入字体色彩库
	color_lib = FontColor(cf.color_path)
	
	# init image_list
	img_root_path = cf.bg_path
	imnames=os.listdir(img_root_path)

	wrong_dict = init_wrong_dict(cf.word_false_txt_path)		
	
	labels_path = os.path.join('./'+cf.output_dir,'labels.txt')
	start = 0
	if os.path.exists(labels_path):  # 支持中断程序后，在生成的图片基础上继续
		with open(labels_path,'r',encoding='utf-8') as f:
			lines = len(f.readlines())
			start =  lines
			print('Resume generating from step %d'%lines)
	
	print('Start generating...')
	print('Saving images in directory : {}'.format(cf.output_dir))
	start_timer = time.time()
	with open(labels_path,'a',encoding='utf-8') as f:

	
		for k in tqdm(range(0,len(ch_list))):
			char = ch_list[k]
			for i in range(cf.num_per_word):
				imname = random.choice(imnames)
				img_path = os.path.join(img_root_path,imname)

				img = Image.open(img_path)
				if img.mode != 'RGB':
					img = img.convert('RGB')

				w, h = img.size

				font_path = random.choice(fonts_list)
				
				font_type = font_path.split('/')[3].split('.')[0]  

				false_words = False

				while font_type in wrong_dict and len(set(char) & set(wrong_dict[font_type])) != 0:
					font_path = random.choice(fonts_list)
					font_type = font_path.split('/')[3].split('.')[0] 

				gen_img, char = generate_single_word(img_path,color_lib,font_path,font_unsupport_chars,char,cf)  


				if gen_img.mode != 'RGB':
					gen_img= gen_img.convert('RGB')

				#高斯模糊
				if cf.blur:
					image_arr = np.array(gen_img) 
					gen_img = apply_blur_on_output(image_arr)            
					gen_img = Image.fromarray(np.uint8(gen_img))
				#模糊图像，模拟小图片放大的效果
				if cf.prydown:
					image_arr = np.array(gen_img) 
					gen_img = apply_prydown(image_arr)
					gen_img = Image.fromarray(np.uint8(gen_img))
				#左右运动模糊
				if cf.lr_motion:
					image_arr = np.array(gen_img)
					gen_img = apply_lr_motion(image_arr)
					gen_img = Image.fromarray(np.uint8(gen_img))       
				#上下运动模糊       
				if cf.ud_motion:
					image_arr = np.array(gen_img)
					gen_img = apply_up_motion(image_arr)        
					gen_img = Image.fromarray(np.uint8(gen_img)) 
				#  apply noise
				if cf.noise and apply(flag.noise):
					gen_img = np.clip(gen_img, 0., 255.)
					gen_img = noiser.apply(gen_img)
					gen_img = Image.fromarray(np.uint8(gen_img))

				save_img_name = "img_{number}.jpg".format(number = str(start+i).zfill(7))
				gen_img_path = os.path.join(cf.output_dir,save_img_name)
				gen_img.save(gen_img_path)
				f.write("{0},{1}\n".format(gen_img_path, char))
				start +=1

	end_timer = time.time()
	hours, rem = divmod(end_timer-start_timer, 3600)
	minutes, seconds = divmod(rem, 60)
	print("It takes {:0>2}:{:0>2}:{:05.2f} to generate.".format(int(hours),int(minutes),seconds))


