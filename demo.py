
import cv2
import os
import time
import paddle.fluid.layers as P
from model.decode_pd import Decode

if __name__ == '__main__':
    file = 'data/voc_classes.txt'
    # file = 'data/coco_classes.txt'

    model_path = 'ep000130-loss3.803-val_loss9.104.pd'
    # model_path = 'aaa'


    # 选一个
    # input_shape = (320, 416)
    input_shape = (416, 416)
    # input_shape = (608, 608)

    use_gpu = True
    # use_gpu = False


    inputs = P.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
    _decode = Decode(inputs, 0.3, 0.45, input_shape, model_path, file, initial_filters=32, use_gpu=use_gpu)

    # detect images in test floder.
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            start = time.time()
            for f in files:
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image = _decode.detect_image(image)
                cv2.imwrite('images/res/' + f, image)
            print('total time: {0:.6f}s'.format(time.time() - start))


