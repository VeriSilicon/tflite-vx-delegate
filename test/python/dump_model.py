import utils
import argparse
import numpy as np
import json
import os
import shutil
import model_cut
import tflite_runtime.interpreter as tflite

print(os.getpid())

## test given model with random input
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        default="/tmp/mobilenet_v1_1.0_224_quant.tflite",
        help = 'model to be compared'
    )
    parser.add_argument(
      '-e',
      '--ext_delegate',
      help='external_delegate_library path'
    )
    parser.add_argument(
      '-d',
      '--dump_location',
      default="/tmp",
      help='location of the model dump file'
    )
    parser.add_argument(
      '-t',
      '--tensor_list',
      default='',
      help="the list of tensor to be dumped, if not supply, all tensor will be dumped"
           "expect a list of number split by comma without space, for example: '16,32,38'"
    )
    args = parser.parse_args()
    with open(args.model, 'rb') as f:
        model_buffer = f.read()
    dump_path = args.dump_location + "/model_dump"
    if os.path.exists(dump_path ):
        shutil.rmtree(dump_path)
    os.makedirs(dump_path + '/cpu')
    os.makedirs(dump_path + '/npu')
    dump_file = open(dump_path + "/summary.txt",'w')

    tensor_list = list()
    if args.tensor_list:
        tensor_list = list(args.tensor_list.split(','))
        tensor_list = [int(i) for i in tensor_list]
    else:
        interpreter = tflite.Interpreter(args.model)
        tensor_list = range(interpreter._interpreter.NumTensors())

    for idx in tensor_list:
        cuted_model = model_cut.buffer_change_output_tensor_to(model_buffer, idx)
        model_path = "/tmp/cutted_model.tflite"
        with open( model_path, 'wb') as g:
            g.write(cuted_model)
        cpu_runner = utils.cpu()
        (gold_input, gold_output) = cpu_runner.run_with_rand_data(model_path) 
        npu_runner = utils.npu(args.ext_delegate)
        npu_output = npu_runner.run(model_path, gold_input)

        gold, npu = gold_output[0], npu_output[0]
        tensor_name = npu[0]
        tensor_name = tensor_name.replace('/', '_')
        tensor_cpu = dump_path + '/cpu/' + tensor_name + '.json'
        tensor_npu = dump_path + '/npu/' + tensor_name + '.json'

        with open(tensor_cpu, 'w') as cf:
            json.dump(gold.tolist(), cf)
        with open(tensor_npu, 'w') as nf:
            json.dump(npu[1].tolist(), nf)
        
        item = "[" + str(idx) +"][" + str(npu[0]) + "] cosine_similarity = " + str(utils.cosine_similarity(gold.flatten(), npu[1].flatten()))
        dump_file.write(item + '\n')
    dump_file.close()
    