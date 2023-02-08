import numpy as np
from numpy import dot
import tflite_runtime.interpreter as tflite
from numpy.linalg import norm

class cpu:
    def __init__(self) -> None:
        # self.ext_delegate = tflite.load_delegate(vx_delegate_lib)
        pass

    def run_with_rand_data(self, model):
        self.interpreter = tflite.Interpreter(model)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()
        in_data = []
        for input in self.input_details:
            idx = input['index']
            shape = input['shape']
            np_dtype = input['dtype']

            data = np.random.normal(0, 127, shape).astype(np_dtype)
            # data = np.zeros(shape).astype(np_dtype)
            self.interpreter.set_tensor(idx, data)
            in_data.append(data)

        self.interpreter.invoke()

        out = []
        for output in self.output_details:
            out.append(self.interpreter.get_tensor(output['index']))

        return (in_data, out)

class npu:
    def __init__(self, vx_delegate_lib) -> None:
        self.ext_delegate = tflite.load_delegate(vx_delegate_lib)

    def run(self, model, input_list):
        self.interpreter = tflite.Interpreter(model, experimental_delegates= [self.ext_delegate])
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()
        len(self.input_details) == len(input_list)

        # TODO order of input from CPU interpreter is aligned to NPU ??
        idx = 0
        for input in self.input_details:
            self.interpreter.set_tensor(input['index'], input_list[idx])
            idx = idx + 1

        self.interpreter.invoke()

        out = []
        for o in self.output_details:
            out.append((o['name'], self.interpreter.get_tensor(o['index'])))
        return out


def norm_ (List1):
    r = 0
    for i in List1:
        r += float(i)*float(i)
    return r
def dot_(L1, L2):
    r = 0
    for (i, j) in zip(L1, L2):
        r += float(i)*float(j)
    return r

def cosine_similarity(List1, List2):
    return dot(List1, List2)/(0.00001+(norm(List1)*norm(List2)))
    #return dot_(List1, List2)/(norm_(List1)*norm(List2))
