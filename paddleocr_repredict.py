import os
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# # 设置GPU环境变量
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 初始化PaddleOCR对象，启用GPU模式
ocr = PaddleOCR()


# 设置输入和输出文件夹的路径
orig_folder = '/home/lmf/mmsegmentation/data/tamper/train2/img'
seg_folder = '/home/lmf/mmsegmentation/val_paddle'
output_folder = '/home/lmf/mmsegmentation/val_paddle_corrected'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有图片文件
image_files = os.listdir(orig_folder)

# 遍历每个图片文件
for image_file in image_files:
    # 获取图片文件的文件名
    filename = os.path.splitext(image_file)[0]

    # 构建输出文件和标记图像文件的路径
    output_file = os.path.join(output_folder, filename + '_corrected.jpg')
    seg_file = os.path.join(seg_folder, filename + '.jpg')

    # 读取原始图像和标记图像
    orig_image = Image.open(os.path.join(orig_folder, image_file))
    seg_image = Image.open(seg_file)

    # 使用PaddleOCR库进行文本检测，获取OCR结果
    result = ocr.ocr(os.path.join(orig_folder, image_file))

    # 创建一个与原始图像大小相同的空白图像（全零数组）来存储OCR结果
    ocr_image = Image.new('RGB', orig_image.size, (0, 0, 0))

    # 将OCR结果绘制到OCR图像中（设置为255）
    for line in result:
        for word in line:
            text = word[1][0]  # 获取文本
            bbox = word[0]  # 获取边界框坐标
            bbox = [int(coord) for point in bbox for coord in point]  # 将边界框坐标转换为整数类型
            bbox = [min(bbox[::2]), min(bbox[1::2]), max(bbox[::2]), max(bbox[1::2])]  # 计算边界框的最小和最大坐标
            ocr_image.paste((255, 255, 255), tuple(bbox), mask=None)

    ocr_image = ocr_image.convert('L')
    corrected_image = np.where(np.array(ocr_image) == 255, seg_image, 0)


    # Save the corrected image
    Image.fromarray(corrected_image.astype(np.uint8)).save(output_file)
