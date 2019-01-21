from yolo_model.yolo import YOLO
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                        output_names, freeze_var_names)

        return frozen_graph

def create_tf():
    f = YOLO()
    from keras import backend as K
    frozen_graph = freeze_session(K.get_session(),
                                    output_names=[out.op.name for out in f.yolo_model.outputs])

    tf.train.write_graph(frozen_graph, '.', "yolo_model.pb", as_text=False)

def test_model():
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        with gfile.FastGFile('yolo_model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            g_in = tf.import_graph_def(graph_def)

            writer = tf.summary.FileWriter("log/")
            writer.add_graph(sess.graph)
            writer.flush()
            writer.close()

if __name__ == "__main__":
    create_tf()
    test_model()
