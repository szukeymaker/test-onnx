"""
删除和新增onnx节点
"""

import onnx
from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto


def add_cast_node(nodes):
    new_nodes = []
    for node in nodes:
        if node.name == "add":
            '''
            new_scale_node = onnx.helper.make_node(
                "Add",
                inputs=['conv_output', 'add_input'],
                outputs=['add_output'],
                name='add')
            '''
            new_add_node = onnx.helper.make_node(
                'Cast',
                inputs=['add'],
                outputs=['output'],
                name='cast',
                to=TensorProto.INT64
            )
            new_nodes += [new_add_node]
        else:
            new_nodes += [node]

    return new_nodes


if __name__ == '__main__':
    model = onnx.load('resize_conv_add.onnx')
    graph = model.graph
    nodes = graph.node
    opset_version = model.opset_import[0].version
    opset_version = 11
    graph_name = f"{graph.name}-int32"
    # new_nodes = delete_add_node(nodes)
    new_nodes = add_cast_node(nodes)
    graph.output[0].type.tensor_type.elem_type = 7
    graph_int32 = h.make_graph(
        new_nodes,
        graph_name,
        graph.input[:-1],
        graph.output,
        initializer=graph.initializer,
    )

    model_int32 = h.make_model(graph_int32, producer_name="onnx-typecast")
    model_int32.opset_import[0].version = opset_version
    ch.check_model(model_int32)
    onnx.save_model(model_int32, "add_cast.onnx")
