import tensorflow as tf
import optimize_for_inference_lib as optimize_for_inference_lib

MODEL_NAME = 'saved_model'

# Freeze the graph

output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    #data =unicode(f.read(), errors='ignore')
    input_graph_def.ParseFromString(f.read())

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["I"], # an array of the input node(s)
        ["O"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph
f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())
