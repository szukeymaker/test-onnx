import onnxruntime
import numpy as np

# 加载模型
weight_path = './resize_conv_add.onnx'
session = onnxruntime.InferenceSession(weight_path) # 加载模型
# 获取输入节点名称
session.get_modelmeta()
input_name = session.get_inputs()[0].name
roi_name = session.get_inputs()[1].name
scales_name = session.get_inputs()[2].name
conv_weight_name = session.get_inputs()[3].name
conv_bias_name = session.get_inputs()[4].name
add_input_name = session.get_inputs()[5].name
output_name = session.get_outputs()[0].name

# 定义输入节点向量
input_data = np.random.randn(1,3,256,256).astype(np.float32)
roi_data = np.array([]).astype(np.float32)
scales_data = np.array([1,1,2,2]).astype(np.float32)
conv_weight_data = np.random.randn(32,3,3,3).astype(np.float32)
conv_bias_data = np.random.randn(32).astype(np.float32)
add_input_data = np.random.randn(1).astype(np.float32)

# 推理
input_dict = {input_name:input_data,roi_name:roi_data,scales_name:scales_data,conv_weight_name:conv_weight_data,
              conv_bias_name:conv_bias_data,add_input_name:add_input_data}
result = session.run(None,input_dict) # 推理模型，输入向量采用字典类型表示
print('result[0]_shape=',result[0].shape)
