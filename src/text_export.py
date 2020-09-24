import datetime
import json
import os.path

class TextExport:
    
    number = 1
    data = {'expressions': []}
    datetime_obj = datetime.datetime.now()
    timestamp = datetime_obj.strftime("%d%b%Y%H%M")
    output_file_path = './logs/output' + str(timestamp) + '.json'

    def __init__(self):
        if os.path.isfile(self.output_file_path):
            self.file = open(self.output_file_path, "a+")
            with open(self.output_file_path) as json_file:
                self.data = json.load(json_file)
        else:
            self.file = open(self.output_file_path, "a+")

    def append(self, frame_number, px, py, expression):
        datetime_obj = datetime.datetime.now()
        timestamp = datetime_obj.strftime("%d.%b.%Y - %H:%M:%S")
        self.data['expressions'].append({
            'number': self.number,
            'frame': frame_number,
            'position': str(px) + str(py),
            'expression': expression,
            'timestamp': timestamp
        })
        self.number += 1

        with open(self.output_file_path, 'w') as outfile:
            json.dump(self.data, outfile, indent=4)

    def close(self):
        self.file.close()


if __name__ == "__main__":
    export = TextExport()
    # parameters -> framenumber, (top,left), (right,bottom), face_expression
    export.append(3, (11, 22), (33, 44), "happy")
    export.close()
