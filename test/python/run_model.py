import utils
import argparse
import numpy as np

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

    args = parser.parse_args()
    
    cpu_runner = utils.cpu()
    (gold_input, gold_output) = cpu_runner.run_with_rand_data(args.model) 
    npu_runner = utils.npu(args.ext_delegate)
    npu_output = npu_runner.run(args.model, gold_input)

    idx = 0
    for (gold, npu) in zip(gold_output, npu_output):
        np.savetxt("/tmp/gold_{}".format(idx), gold.flatten())
        np.savetxt("/tmp/npu_{}".format(idx), npu.flatten())

        print("[{}]cosine_similarity = ".format(idx), utils.cosine_similarity(gold.flatten(), npu.flatten()))
        idx = idx + 1