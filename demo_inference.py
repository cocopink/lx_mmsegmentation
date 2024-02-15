import os
import glob
from PIL import Image
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor

def load_model(config_file, checkpoint_file):
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    return model

def process_image(model, image):
    numpy_image = np.array(image)
    result = inference_segmentor(model, numpy_image)
    return result

config_file = '/home/lmf/mmsegmentation/work_configs/tamper/tamper_convx_l.py'
checkpoint_file = '/home/lmf/mmsegmentation/work_dirs/tamper/convx_l_12x_dice_aug1_dec/epoch_144.pth'
model = load_model(config_file, checkpoint_file)
model.eval()

# 设置输入和输出文件夹
input_folder = '/home/lmf/mmsegmentation/data/tamper/train2/img'
output_folder = '/home/lmf/mmsegmentation/val_paddle'

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹下的所有图片
image_files = glob.glob(os.path.join(input_folder, '*'))

image = Image.open('/home/lmf/mmsegmentation/0000.jpg')#image_file
result = process_image(model, image)

# 假设result是你的分割结果
seg_map = result[0]

# 篡改区域设置为255，真实区域设置为0
seg_map[seg_map == 1] = 255
seg_map[seg_map == 0] = 0

# 转换为PIL图像
seg_img = Image.fromarray(seg_map.astype(np.uint8))

# # 获取图像的文件名
# basename = os.path.basename(image_file)
# # 为输出文件构建路径
# output_file = os.path.join(output_folder, basename)

# 保存图像
seg_img.save('/home/lmf/mmsegmentation/0000_changed.png')#output_file

# 对每一张图片进行处理
# for image_file in image_files:
#     image = Image.open('/home/lmf/0628.jpg')#image_file
#     result = process_image(model, image)

#     # 假设result是你的分割结果
#     seg_map = result[0]

#     # 篡改区域设置为255，真实区域设置为0
#     seg_map[seg_map == 1] = 255
#     seg_map[seg_map == 0] = 0

#     # 转换为PIL图像
#     seg_img = Image.fromarray(seg_map.astype(np.uint8))

#     # 获取图像的文件名
#     basename = os.path.basename(image_file)
#     # 为输出文件构建路径
#     output_file = os.path.join(output_folder, basename)
    
#     # 保存图像
#     seg_img.save('/home/lmf/0628.png')#output_file
    
