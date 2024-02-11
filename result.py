import pickle
import cv2

with open('results_new.pkl', 'rb') as f:
    results = pickle.load(f)
for i, value in enumerate(results):
    # 获取预测结果
    if isinstance(value, dict):
        pred = value['pred']
    elif isinstance(value, list):
        pred = value[0]['pred']
    else:
        pred = value[0].data.cpu().numpy().argmax(axis=0)

    # 处理预测结果
    image = pred.astype('uint8') * 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f'result_{i}.png', image)