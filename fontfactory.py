import os
import hashlib
import pickle
from fontTools.ttLib import TTCollection, TTFont
class FontFactory():

	def __init__(self,charset,fonts_path):
		self.charset = charset

		font_files = os.listdir(fonts_path)
		fonts_list=[]
		for font_file in font_files:
			font_path=os.path.join(fonts_path,font_file)
			fonts_list.append(font_path)

		self.fonts_list = fonts_list

	def load_font(self,font_path):
		"""
		Read ttc, ttf, otf font file, return a TTFont object
		"""

		# ttc is collection of ttf
		if font_path.endswith('TTC'):
			ttc = TTCollection(font_path)
			# assume all ttfs in ttc file have same supported chars
			return ttc.fonts[0]

		if font_path.endswith('ttf') or font_path.endswith('TTF') or font_path.endswith('otf'):
			ttf = TTFont(font_path, 0, allowVID=0,
						 ignoreDecompileErrors=True,
						 fontNumber=-1)

			return ttf

	def md5(self,string):
		m = hashlib.md5()
		m.update(string.encode('utf-8'))
		return m.hexdigest()


	def check_font_chars(self,ttf, charset):
		"""
		Get font supported chars and unsupported chars
		:param ttf: TTFont ojbect
		:param charset: chars
		:return: unsupported_chars, supported_chars
		"""
		#chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)
		chars_int=set()
		for table in ttf['cmap'].tables:
			for k,v in table.cmap.items():
				chars_int.add(k)            
				
		unsupported_chars = []
		supported_chars = []
		for c in charset:
			if ord(c) not in chars_int:
				unsupported_chars.append(c)
			else:
				supported_chars.append(c)

		ttf.close()
		return unsupported_chars, supported_chars


	def get_fonts_chars(self):
		"""
		loads/saves font supported chars from cache file
		:param fonts: list of font path. e.g ['./data/fonts/msyh.ttc']
		:param chars_file: arg from parse_args
		:return: dict
			key -> font_path
			value -> font supported chars
		"""
		out = {}

		cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', '.caches'))
		if not os.path.exists(cache_dir):
			os.makedirs(cache_dir)

		chars = ''.join(self.charset)

		for font_path in self.fonts_list:
			string = ''.join([font_path, chars])
			file_md5 = self.md5(string)

			cache_file_path = os.path.join(cache_dir, file_md5)
			print(cache_file_path)
			if not os.path.exists(cache_file_path):
				ttf = self.load_font(font_path)
				_, supported_chars = self.check_font_chars(ttf, chars)
				print('Save font(%s) supported chars(%d) to cache' % (font_path, len(supported_chars)))

				with open(cache_file_path, 'wb') as f:
					pickle.dump(supported_chars, f, pickle.HIGHEST_PROTOCOL)
			else:
				with open(cache_file_path, 'rb') as f:
					supported_chars = pickle.load(f)
				print('Load font(%s) supported chars(%d) from cache' % (font_path, len(supported_chars)))

			out[font_path] = supported_chars

		return out

	def get_unsupported_chars(self):
		"""
		Get fonts unsupported chars by loads/saves font supported chars from cache file
		:param fonts:
		:param chars_file:
		:return: dict
			key -> font_path
			value -> font unsupported chars
		"""
		charset = ''.join(self.charset)
		fonts_chars = self.get_fonts_chars()
		fonts_unsupported_chars = {}
		for font_path, chars in fonts_chars.items():
			unsupported_chars = list(filter(lambda x: x not in chars, charset))
			fonts_unsupported_chars[font_path] = unsupported_chars
		return fonts_unsupported_chars

	def get_fonts_list(self):
		return self.fonts_list
