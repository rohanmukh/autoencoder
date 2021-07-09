import time
import boto3
import sys
from collections import defaultdict
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from google.protobuf import text_format
import tensorflow as tf
import traceback
from tensorflow.python.eager.def_function import Function

models=["example1", "example2", "example4"]
zoo_metadata={
        "example1": (1, 28, 28, 1, "input_1", "float32"),
        "example2": (1, 28, 28, 1, "input_1", "float32"),
        "example4": (1, 28, 28, 1, "input_1", "float32"),
        }

from tensorflow.python.tools.saved_model_utils import get_saved_model_tag_sets



def get_output_names(tf_graph):
    output_tensor_names = []
    for tensor in tf_graph.outputs:
        output_tensor_names.append(tensor.name)
    return output_tensor_names
    

for model_name in models:
    bs, w, h, c, input_name,data_type = zoo_metadata[model_name]
    inshape = [bs]
    for dim in [w,h,c]:
        if dim is not None:
            inshape.append(dim)
    inshape = tuple(inshape)
    if data_type == "float32":
        tf_in_type = tf.float32
    else:
        tf_in_type = tf.uint8

    input_x = np.random.uniform(0.0, 255.0, size=inshape).astype(data_type)/255
    
    file_name = model_name + '/saved_trt_model' 
    loaded = tf.saved_model.load(file_name)
    #import pdb; pdb.set_trace()
    if len(loaded.signatures) == 0:
        f = loaded.__call__.get_concrete_function(tf.TensorSpec(inshape, dtype=tf_in_type))
    elif 'serving_default' in loaded.signatures:
        f = loaded.signatures['serving_default']
    else:
        f = loaded.signatures[list(loaded.signatures.keys())[0]]
    frozen_func = convert_variables_to_constants_v2(f, lower_control_flow=False)
    #tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=".", name="frozen_graph.pb",as_text=True)
    tf_graph = frozen_func.graph.as_graph_def(add_shapes=True)
    output_tensor_names = get_output_names(frozen_func)
    print("*******************************")

    for i in range(1000):
        #input_x = tf.convert_to_tensor(input_x)
        tf_out = loaded(input_x).numpy()
    
    stime = time.time()
    for i in range(1000):
        #input_x = tf.convert_to_tensor(input_x)
        tf_out = loaded(input_x).numpy()
    etime = time.time() 
    tf_time = etime - stime


    print("TF time is {:.2f}".format(tf_time))

    

