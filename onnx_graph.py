import onnx

model = onnx.load("distilbert_model.onnx")

print(f"Total nodes: {len(model.graph.node)}")

op_types={}
for node in model.graph.node:
    op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    
print("\nOperations breakdown:")
for op, count in sorted(op_types.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count}")
    
print("\n Fusion candidates :")
for i in range(len(model.graph.node)-1):
    current = model.graph.node[i].op_type
    next = model.graph.node[i+1].op_type
    if current in ['Add', 'Mul', 'MatMul'] and next in ['Relu', 'Gelu', 'LeakyRelu', 'Tanh', 'Softmax']:
        print(f"  {current} -> {next}")