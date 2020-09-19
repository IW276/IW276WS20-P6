import datetime
import json
import os.path


class TextExport:
    output_file = ''
    number = 1
    data = {'expressions': []}

    def __init__(self, path_and_file_name):
        self.output_file = path_and_file_name
        if os.path.isfile(path_and_file_name):
            self.file = open(path_and_file_name, "a+")
            with open(path_and_file_name) as json_file:
                self.data = json.load(json_file)
        else:
            self.file = open(path_and_file_name, "a+")

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

        with open(self.output_file, 'w') as outfile:
            json.dump(self.data, outfile, indent=4)

    def close(self):
        self.file.close()


if __name__ == "__main__":
    export = TextExport("test.json")
    export.append(3, (11, 22), (33, 44), "happy")
    export.close()
    # framenumber,(top,left),(right,bottom),face_expression
