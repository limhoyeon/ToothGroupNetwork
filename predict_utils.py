import glob
import json
import os
import numpy as np
import traceback


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ScanSegmentation():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self, model):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.chl_pipeline = model

        #self.model = load_model()
        #sef.device = "cuda"

        pass

    @staticmethod
    def load_input(input_dir):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """

        # iterate over files in input_dir, assuming only 1 file available
        inputs = glob.glob(f'{input_dir}/*.obj')
        print("scan to process:", inputs)
        return inputs

    @staticmethod
    def write_output(labels, instances, jaw, output_path):
        """
        Write to /output/dental-labels.json your predicted labels and instances
        Check https://grand-challenge.org/components/interfaces/outputs/
        """
        pred_output = {'id_patient': "",
                       'jaw': jaw,
                       'labels': labels,
                       'instances': instances
                       }

        # just for testing
        #with open('./test/test_local/expected_output.json', 'w') as fp:
        with open(output_path, 'w') as fp:
            json.dump(pred_output, fp, cls=NpEncoder)

        return

    @staticmethod
    def get_jaw(scan_path):
        try:
            # read jaw from filename
            _, jaw = os.path.basename(scan_path).split('.')[0].split('_')
        except:
            # read from first line in obj file
            try:
                with open(scan_path, 'r') as f:
                    jaw = f.readline()[2:-1]
                if jaw not in ["upper", "lower"]:
                    return None
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())
                return None

        return jaw

    def predict(self, inputs):
        """
        Your algorithm goes here
        """

        try:
            assert len(inputs) == 1, f"Expected only one path in inputs, got {len(inputs)}"
        except AssertionError as e:
            raise Exception(e.args)
        scan_path = inputs[0]
        #print(f"loading scan : {scan_path}")
        # read input 3D scan .obj
        try:
            # you can use trimesh or other any loader we keep the same order
            #mesh = trimesh.load(scan_path, process=False)
            pred_result = self.chl_pipeline(scan_path)
            jaw = self.get_jaw(scan_path)
            if jaw == "lower":
                pred_result["sem"][pred_result["sem"]>0] += 20
            elif jaw=="upper":
                pass
            else:
                raise "jaw name error"
            print("jaw processed is:", jaw)
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            raise
        # preprocessing if needed
        # prep_data = preprocess_function(mesh)
        # inference data here
        # labels, instances = self.model(mesh, jaw=None)

        # extract number of vertices from mesh
        nb_vertices = pred_result["sem"].shape[0]

        # just for testing : generate dummy output instances and labels
        instances = pred_result["ins"].astype(int).tolist()
        labels = pred_result["sem"].astype(int).tolist()

        try:
            assert (len(labels) == len(instances) and len(labels) == nb_vertices),\
                "length of output labels and output instances should be equal"
        except AssertionError as e:
            raise Exception(e.args)

        return labels, instances, jaw

    def process(self, input_path, output_path):
        """
        Read input from /input, process with your algorithm and write to /output
        assumption /input contains only 1 file
        """
        #input = self.load_input(input_dir='./test/test_local')
        labels, instances, jaw = self.predict([input_path])
        self.write_output(labels=labels, instances=instances, jaw=jaw, output_path=output_path)