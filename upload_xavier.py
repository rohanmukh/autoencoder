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
        "example1": (1, 28, 28, 3, "input_1", "float32"),
        "example2": (1, 28, 28, 3, "input_1", "float32"),
        "example4": (1, 28, 28, 3, "input_1", "float32"),
        }

from tensorflow.python.tools.saved_model_utils import get_saved_model_tag_sets

def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy()]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


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
    
    file_name = model_name + '/saved_model' 
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


    import tvm
    from tvm import relay
    from tvm import runtime
    import tvm.relay.testing

    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

    try:
        from tvm.relay.frontend import tensorflow2
        mod, params = tensorflow2.from_tensorflow(tf_graph,shape={input_name: inshape}, outputs=output_tensor_names)
    except:
        e = sys.exc_info()
        print(model_name)
        print(e)
        traceback.print_exc()
        continue
    print("**************frontend passed***************")

    mod, config = partition_for_tensorrt(mod, params, remove_no_mac_subgraphs=True)
            #use_implicit_batch=False)
    mod = tvm.relay.transform.InferType()(mod)
    with tvm.transform.PassContext(
           opt_level=3, 
           config={'relay.ext.tensorrt.options': config}, 
           disabled_pass=["FoldScaleAxis"]
           ):
   # with relay.build_config(opt_level=3):
        set_cuda_target_arch("sm_72")
        #vm_exec = relay.vm.compile(mod, target='cuda -libs=thrust', params=params)
        vm_exec = relay.vm.compile(mod, target='cuda -libs=thrust -arch=sm_72', target_host='llvm -mtriple=aarch64-linux-gnu', params=params)
    print("compilation done")
    code, lib = vm_exec.save()
    lib.export_library(
        model_name + '/' + model_name+'.so',
        #cc='/usr/bin/g++',
        cc='/usr/bin/aarch64-linux-gnu-g++',
    )
    with open(model_name + '/' + model_name+'.code', "wb") as fo:
        fo.write(code)
    print("export compilation")
    
    #vm_exec = tvm.runtime.vm.Executable.load_exec(code, lib)

    #des_vm = tvm.runtime.vm.VirtualMachine(vm_exec, tvm.cuda(0))
    #
    ##Warmup
    #for i in range(1000):
    #    tvm_out = des_vm.invoke('main', **{input_name:input_x})
    #
    #stime = time.time() 
    #for i in range(1000):
    #    tvm_out = des_vm.invoke('main', **{input_name:input_x})
    #etime = time.time() 
    #tvm_time = etime - stime

    #speedup = tf_time/tvm_time
    #print("Speedup is {:.2f}".format(speedup))
    #tvm_out = vmobj_to_list(tvm_out)[0]

    #
    #tvm.testing.assert_allclose(tf_out, tvm_out, rtol=5e-3, atol=5e-3)
    #print("Accuracy test successful")

