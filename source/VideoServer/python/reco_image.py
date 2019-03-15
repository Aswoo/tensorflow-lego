import os, sys
import cv2
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def recognition_image(img,width,height):

    
    np_image_data = np.asarray(img)
    #np_image_data = np_image_data.reshape(1, 224 * 224 * 3)    
#np_image_data = cv2.resize(np_image_data,(width,height), interpolation = cv2.INTER_CUBIC)
    # maybe insert float convertion here - see edit remark!
    np_final = np.expand_dims(np_image_data, axis=0)
    
    #tfImage = np.array(img)[:, :, 0:3]
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'Mul:0': np_final})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
    return top_k
