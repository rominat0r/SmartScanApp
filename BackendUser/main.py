import os
import chime
import cv2 
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from product_list import ext_labels
import logging
 
logging.basicConfig(filename = '/var/www/html/BackendUser/errors.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')


                    

print('Runninng script...')
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

configs = config_util.get_configs_from_pipeline_file('/var/www/html/BackendUser/'+files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('/var/www/html/BackendUser/'+paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def scan(imgname):
#-------------------------------------------------------------
    # IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
    # cap = cv2.VideoCapture(2,cv2.CAP_V4L)
    # result = True
    # while(result):
    #     ret, frame = cap.read()
    #     imgname = os.path.join(IMAGES_PATH,'test'+'.'+'{}.jpg'.format(str(uuid.uuid1())))
    #     cv2.imwrite(imgname, frame)
    #     result = False
    # cap.release()
    # chime.success()
    # cv2.destroyAllWindows()
#-------------------------------------------------------------

    category_index = label_map_util.create_category_index_from_labelmap('/var/www/html/BackendUser/'+files['LABELMAP'])
    IMAGES_PATH = os.path.join('/var/www/html/BackendUser/mediafiles/posts/')
    print(imgname)
    imgpath = os.path.join(IMAGES_PATH, imgname)
    img = cv2.imread(imgpath)
    image_np = np.array(img)
    chime.success()
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
# detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    products = []

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)
    for a in np.argwhere(detections['detection_scores'] > 0.9):
        product  = np.array([category_index[np.array(detections['detection_classes'])[a[0]]+1]['id']])

        #products[str(product[0])] = ext_labels[str(product[0])]
        products.append(ext_labels[str(product[0])])
        
        #products.update({  ext_labels[product[0]]  })
    
    return products




