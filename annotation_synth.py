# -*- coding:utf-8 -*-

import os
import csv
import numpy as np
import cv2
import shutil
import random
from os.path import join as pjoin
import json

SRC = "./img_src/"
SRC_BG = "./nop_indoor_img/"
DST = "./img_dst/"
skip_list = []
skipbg_list = []
global sync_cnt
target_cnt = 5

state = 0

global label
enable_debug = False
def rad(x):
    return x*np.pi/180

def getDocSize(path):
    try:
        size = os.path.getsize(path)
        return formatSize(size)
    except Exception as err:
        print(err)
        exit()

def check_dir(path):
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)


def writeFile(filename):
    global label

    f = open(filename, 'w')
    iter = 0
    label_cnt = 0
    for e in label:
        if iter==8:
            f.write(str(e)+"\n")
            iter=0
            label_cnt = label_cnt + 1
        else:
            f.write(str(e)+",")
            iter=iter+1
    label = []
    card_num = []
    f.close()

def synth_img(fg_image_path, bg_image_path, combine_title):
    global label, sync_cnt
    fg_img_ori = cv2.imread(fg_image_path)
    bg_img_ori = cv2.imread(bg_image_path)

    random_shape=np.array([0, 1, 2, 3, 4])
    rd_shape = np.random.choice(random_shape)
    if rd_shape<4:
        bg_height = 640
        bg_width = 480
    else:
        bg_height = 480
        bg_width = 640
    bg_img = cv2.resize(bg_img_ori, dsize=(int(bg_width), int(bg_height)))
    bg_height,bg_width=bg_img.shape[0:2]

    fg_ori_height,fg_ori_width=fg_img_ori.shape[0:2]
    fg_scale = random.uniform(2/3, 1/6)
    fg_width = int(fg_scale*bg_width)
    fg_img = cv2.resize(fg_img_ori, dsize=(int(fg_width), int(fg_ori_height*fg_width/fg_ori_width)))
    fg_height,fg_width=fg_img.shape[0:2]

    dx_left = 0
    dx_right = 0
    dy_up = 0
    dy_down = 0
    if bg_width>fg_width and bg_height>fg_height:
        left_ratio = random.uniform(0.0, 8.0)
        up_ratio = random.uniform(0.0, 8.0)
        dx_left = left_ratio*(bg_width-fg_width)
        dx_right = (1-left_ratio)*(bg_width-fg_width)
        dy_up = up_ratio*(bg_height-fg_height)
        dy_down = (1-up_ratio)*(bg_height-fg_height)
    else:
        print('ERROR: bg image too small!...should not happen')
        return False

    anglex = random.uniform(-15, 15)
    angley = random.uniform(-15, 15)
    anglez = random.uniform(-15, 15)
    fov = 50
    z_value = random.randint(-10, 0)
	

    z=np.sqrt(bg_width**2 + bg_height**2)/2/np.tan(rad(fov/2))

    rx = np.array([[1,                  0,                          0,                          0],
                   [0,                  np.cos(rad(anglex)),        -np.sin(rad(anglex)),       0],
                   [0,                 -np.sin(rad(anglex)),        np.cos(rad(anglex)),        0,],
                   [0,                  0,                          0,                          1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0,                         np.sin(rad(angley)),       0],
                   [0,                   1,                         0,                          0],
                   [-np.sin(rad(angley)),0,                         np.cos(rad(angley)),        0,],
                   [0,                   0,                         0,                          1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)),      0,                          0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)),      0,                          0],
                   [0,                  0,                          1,                          0],
                   [0,                  0,                          0,                          1]], np.float32)

    r = rx.dot(ry).dot(rz)

    #4 pair vertex gen
    pcenter = np.array([int(bg_width/2), int(bg_height/2), 0, 0], np.float32)
    
    p1 = np.array([(bg_width-fg_width)/2,(bg_height-fg_height)/2,  z_value,0], np.float32) - pcenter
    p2 = np.array([(bg_width+fg_width)/2,(bg_height-fg_height)/2,  z_value,0], np.float32) - pcenter
    p3 = np.array([(bg_width-fg_width)/2,(bg_height+fg_height)/2,  z_value,0], np.float32) - pcenter
    p4 = np.array([(bg_width+fg_width)/2,(bg_height+fg_height)/2,  z_value,0], np.float32) - pcenter
    
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0,0],
                    [fg_width,0],
                    [0,fg_height],
                    [fg_width,fg_height]], np.float32)
    
    dst = np.zeros((4,2), np.float32)

    #projection transform
    for i in range(4):
        dst[i,0] = list_dst[i][0]*z/(z-list_dst[i][2]) + pcenter[0]
        dst[i,1] = list_dst[i][1]*z/(z-list_dst[i][2]) + pcenter[1]
    min_x = min(min(dst[0,0], dst[1, 0]), min(dst[2, 0], dst[3, 0]))
    max_x = max(max(dst[0,0], dst[1, 0]), max(dst[2, 0], dst[3, 0]))
    min_y = min(min(dst[0,1], dst[1, 1]), min(dst[2, 1], dst[3, 1]))
    max_y = max(max(dst[0,1], dst[1, 1]), max(dst[2, 1], dst[3, 1]))
    minx_res = min_x
    maxx_res = bg_width - max_x
    miny_res = min_y
    maxy_res = bg_height - max_y
    all_points_x = []
    all_points_y = []
    if minx_res>0 and maxx_res>0 and miny_res >0 and maxy_res > 0:
        warpR = cv2.getPerspectiveTransform(org, dst)
        im_out = cv2.warpPerspective(fg_img, warpR, (bg_img.shape[1],bg_img.shape[0]))# flags = cv2.INTER_NEAREST,borderMode=cv2.BORDER_TRANSPARENT)
        for i in range(bg_img.shape[1]):
            for j in range(bg_img.shape[0]):
                pixel = im_out[j, i]
                if not np.all(pixel == [0, 0, 0]):
                    bg_img[j, i] = im_out[j, i]
        #bg_img.save(DST + combine_title +".jpg")
        sync_img_path = DST + str(sync_cnt) +".jpg"
        cv2.imwrite(sync_img_path, bg_img)
        img_file_size = os.path.getsize(sync_img_path)

        #projection
        list_info = dst[0,0], dst[0,1], dst[1,0], dst[1,1],dst[3,0], dst[3,1],dst[2,0], dst[2,1]
        #print(list_info)

        for i in range(4):
            all_points_x.append(int(dst[i,0]))
            all_points_y.append(int(dst[i,1]))
        sync_cnt = sync_cnt + 1
        #filename = str(sync_cnt)
		
        return True, all_points_x, all_points_y, img_file_size
    else:
        print('bad warpPerspective...drop...')
        exit()
        return False, all_points_x, all_points_y, 0

def process(src_path, current_img, src_bg_path, current_bgimg, DST):
    global sync_cnt

    print('processing:')
    print(current_img)
    print(current_bgimg)
    fg_image_path = os.path.join(src_path, current_img)
    bg_image_path = os.path.join(src_bg_path, current_bgimg)
    (fgtitle, ext) = os.path.splitext(current_img)
    (bgtitle, ext) = os.path.splitext(current_bgimg)
    fg_label_file = fgtitle + '.txt'
    combine_title = fgtitle + '_' + bgtitle

    synth_flag, pointx_list, pointy_list, img_file_size = synth_img(fg_image_path, bg_image_path, combine_title)
    if not synth_flag:
        print('synth_img error while processing:')
        print(fg_image_path)
        print(bg_image_path)
        return False
    else:
        json_content = '"'+str(sync_cnt-1)+'.jpg'+str(img_file_size)+'":{"fileref":"","size":'+str(img_file_size)+',"filename":"'+str(sync_cnt-1)+'.jpg'+ '","base64_img_data":"","file_attributes":{},"regions":{"0":{"shape_attributes":{"name":"polygon","all_points_x":['+ str(pointx_list[0])+','+str(pointx_list[1])+','+str(pointx_list[2])+','+str(pointx_list[3])+'],"all_points_y":['+ str(pointy_list[0])+','+str(pointy_list[1])+','+str(pointy_list[2])+','+str(pointy_list[3])+']},"region_attributes":{"class":"'+str(1)+'"}}}},'
        print(json_content)
        fr = open(pjoin(DST, 'annotation.json'), 'a')
        fr.write(json_content)
        fr.close()
    return True

if __name__ == '__main__':

    global current_img_name, label
    global sync_cnt
    check_dir(SRC)
    check_dir(SRC_BG)

    label=[]
    image_file_list = os.listdir(SRC)
    bgimage_file_list = os.listdir(SRC_BG)

    x1, y1, x2, y2 = 0, 0, 0, 0  # bbox coordinate
    scale_index = 0
    sync_cnt = 0
    status = True
    fg_cnt = 0
    bg_cnt = 0

    for fg_cnt in range(0, len(image_file_list)):
        if sync_cnt>=target_cnt:
            exit()
        image_file = image_file_list[fg_cnt]
        for bg_cnt in range(0, 2):#bg count per samples...
            #print(fg_cnt)
            #print(bg_cnt)
            bgimage_file = bgimage_file_list[random.randint(0, len(bgimage_file_list)-1)]
            current_img = image_file
            current_bgimg = bgimage_file
            # main process
            status = process(SRC, current_img, SRC_BG, current_bgimg, DST)
            
            if status == True:
                continue
            else:
                print('ERROR occurs while processing:')
                print(current_img)
                print(current_bgimg)
                exit()