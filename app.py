
import streamlit as st
from PIL import Image
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
import warnings
# 设置页面宽度和高度
st.set_page_config(layout="wide")


# 页面标题和Logo
st.sidebar.markdown("<h1 style='text-align: center; color: #2196F3;'>文本图像篡改检测</h1>", unsafe_allow_html=True)
logo_image = Image.open("logo.png")
st.sidebar.image(logo_image, use_column_width=True)

# 页面导航栏
nav_selection = st.sidebar.radio("导航", ["演示"])

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

warnings.filterwarnings("ignore", category=UserWarning)


@st.cache(allow_output_mutation=True)
def load_model(config_file, checkpoint_file):
    model = init_segmentor(config_file, checkpoint_file, device='cuda')
    return model



config_file = 'tamper_convx_l.py'
checkpoint_file = 'epoch_144.pth'
model = load_model(config_file, checkpoint_file)
model.eval()

def process_image(model, image):
    numpy_image = np.array(image)
    result = inference_segmentor(model, numpy_image)
    return result



# 演示页面
if nav_selection == "演示":
    st.title("")
    st.markdown("<h2 style='text-align: center; color: blue;'>演示页面</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: green;'>选择一张图片进行篡改检测或上传您自己的图片</h3>", unsafe_allow_html=True)

    # 图片选择
    image_options = ["图片1", "图片2", "图片3"]
    selected_image = st.selectbox("选择一张图片", image_options)

    # 创建两列
    col1, col2 = st.columns([1, 1])

    # 本地上传
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 用户上传了图片
        uploaded_image = Image.open(uploaded_file)
        with col1:
            st.image(uploaded_image, caption="已上传的图片", width=700)
    else:
        # 用户没有上传图片，显示默认图片
        if selected_image == "图片1":
            demo_image = Image.open("1.jpg")
        elif selected_image == "图片2":
            demo_image = Image.open("0013.jpg")
        elif selected_image == "图片3":
            demo_image = Image.open("7948.jpg")

        with col1:
            st.image(demo_image, caption="已选择的图片", width=700)

    # 检测按钮
    detect_button = st.button("检测篡改")
    if detect_button:
        # 执行图像篡改检测
        # 如果用户已上传图像，使用上传的图像，否则使用选中的图像
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = demo_image
        # 假设检测后的结果图片为1.png
        result = process_image(model, image)
        seg_map = result[0]

        # 篡改区域设置为255，真实区域设置为0
        seg_map[seg_map == 1] = 255
        seg_map[seg_map == 0] = 0

        # 转换为PIL图像
        seg_img = Image.fromarray(seg_map.astype(np.uint8))
        with col2:
            st.image(seg_img, caption="篡改检测结果", width=700)


