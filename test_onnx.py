import onnx
from onnx import helper
from onnx import TensorProto

# define tensor
input1 = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 256, 256])
roi = helper.make_tensor_value_info('roi', TensorProto.FLOAT, [])
scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
conv_input = helper.make_tensor_value_info('conv_input', TensorProto.FLOAT, [1, 3, 512, 512])
conv_weight = helper.make_tensor_value_info('conv_weight', TensorProto.FLOAT, [32, 3, 3, 3])
conv_bias = helper.make_tensor_value_info('conv_bias', TensorProto.FLOAT, [32])
conv_output = helper.make_tensor_value_info('conv_output', TensorProto.FLOAT, [1, 32, 512, 512])
add_input = helper.make_tensor_value_info('add_input', TensorProto.FLOAT, [1])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 512, 512])

# makenode
resize_node = helper.make_node("Resize", ['input', 'roi', 'scales'], ['conv_input'], name='resize')
conv_node = helper.make_node("Conv", ['conv_input', 'conv_weight', 'conv_bias'], ['conv_output'], name='conv')
add_node = helper.make_node('Add', ['conv_output', 'add_input'], ['output'], name='add')

# makegraph
graph = helper.make_graph([resize_node, conv_node, add_node], 'resize_conv_add_graph',
                          inputs=[input1, roi, scales, conv_weight, conv_bias, add_input], outputs=[output])

# makemodel
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# checkmode
onnx.checker.check_model(model)

# print(model)
onnx.save(model, 'resize_conv_add.onnx')
