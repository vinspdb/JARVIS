from prepocessing.rgb_image import ReadLog
import sys
import os
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    l = sys.argv[1]
    try:
        os.makedirs("img/"+l)
        os.makedirs("fold/"+l)
        get_log = ReadLog(l).readView()
    except FileExistsError:
        pass