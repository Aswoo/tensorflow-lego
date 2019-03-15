import ctypes
from ctypes import cdll
import sys
import numpy as np
import cv2
import os
import json
import my_mysql
import my_cpp_lib
import requests
import datetime
import time
from PIL import Image
from io import BytesIO
import paho.mqtt.client as paho
from functools import partial
import threading
import reco_image



def start_ar_server(mysql_client, mqtt_client, mqtt_topic, edge_port, width, height, lib, CHUNK):
	ar_server = lib.ar_server_new()
	video_handler = lib.video_handler_new()

	lib.ar_server_init(ar_server, int(edge_port))
	lib.video_handler_init(video_handler, width, height)

	buf = (ctypes.c_ubyte * CHUNK)()
	buf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))

	flag = ctypes.c_int()
        
	pkt_buf = ctypes.POINTER(ctypes.c_ubyte)()
	pkt_len = ctypes.c_int()

	print("start while")
	while True:
		lib.ar_server_accept(ar_server)
		#start manager

		traffic_start_time = datetime.datetime.now()
		traffic = 0
		
		is_video_process = [False]
		while True:
			read_len = lib.ar_server_read(ar_server, buf, CHUNK)
			if read_len < 0:
				print("ar_server_read error. read len is " + str(read_len))
				break
			
			lib.pkt_check(buf, read_len, ctypes.byref(flag), ctypes.byref(pkt_buf), ctypes.byref(pkt_len))
			if flag.value == 0:
				np_video_decoded = np.zeros(int(width * height * 3 / 2), dtype=np.uint8)
				video_decoded = np_video_decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
				if lib.video_handler_get_decoded_pkt(video_handler, pkt_buf, pkt_len, ctypes.byref(video_decoded)) == True:
					if is_video_process[0] == False:
						threading.Thread(target=image_recognition_in_video, args=(np_video_decoded, width, height, is_video_process, lib, ar_server, mqtt_client,mqtt_topic)).start()
				else :
					print("video packet not yet")

def image_recognition_in_video(yuv_frame, width, height, is_video_process, lib, ar_server, client,mqtt_topic):
	print("START!")
	is_video_process[0] = True

	yuv_frame = yuv_frame.reshape((int)(height * 3 / 2), width)
	rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV420p2RGB)
	#print("frame")
	rgb_small_frame = cv2.resize(rgb_frame,dsize=(299,299), interpolation = cv2.INTER_CUBIC)

	print("before tensorflow");
	data = reco_image.recognition_image(rgb_small_frame,width,height)
	print("after tensorflow");
	#run tensorflow
	first_data = data[0]
	client.publish(mqtt_topic,int(first_data))
	print(first_data)
	is_video_process[0] = False

def initialize():
	edge_port = 5678
	#we need to initialize this values
	width = int(1280)
	height = int(720)
	#get cpp libraries and chunk size
	lib = my_cpp_lib.get_lib()
	CHUNK = my_cpp_lib.CHUNK

	#load mysql
	mysql_client = my_mysql.MyMysql(os.environ.get('MYSQL_HOST', None), os.environ.get('MYSQL_USER', None), os.environ.get('MYSQL_PWD', None), os.environ.get('MYSQL_DB', None))
	#load mqtt
	mqtt_client = paho.Client(os.environ.get('MQTT_ID', None))
	mqtt_client.connect(os.environ.get('MQTT_IP', None), int(os.environ.get('MQTT_PORT', None)))
	print("mqtt_connet_finish");
	mqtt_topic = os.environ.get('MQTT_TOPIC', None)
	print("mqtt_topic_get: ",mqtt_topic);

	return mysql_client, mqtt_client, mqtt_topic, edge_port, width, height, lib, CHUNK
	
def main():
	mysql_client, mqtt_client, mqtt_topic, edge_port, width, height, lib, CHUNK = initialize()
	start_ar_server(mysql_client, mqtt_client, mqtt_topic, edge_port, width, height, lib, CHUNK)

if __name__ == "__main__":
	main()
