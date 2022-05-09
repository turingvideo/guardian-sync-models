import json
import re
import logging
import os
import copy
from collections import namedtuple


def dict_deep_update(d: dict, new_dict: dict):
    """Recursively update a dictionary.

    This function will iterate over the keys in the 'new_dict' object and
    update the corresponding field in the dictionary 'd'. For fields that
    have dictionary type or other mapping types, the function will recursively
    update the fields. For other data types, it will do a deep copy.

    Notice that this function is designed to simplify the configuration updates,
    so both the 'd' and 'new_dict' usually only contain built-in types such as
    numeric, tuple, list, dict, etc. For instance values, the deep copy may
    result in unexpected behavior.

    Args:
        d (dict): The dictionary to be updated.
        new_dict (mapping): The mapping object contains the updates.

    Returns:
        The updated dictionary.

    """
    for k, v in new_dict.items():
        if isinstance(v, dict):
            if k not in d:
                d[k] = copy.deepcopy(v)
            else:
                d[k] = dict_deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def update_config(configs: dict, config_file_path: str):
    """Update existing configurations with json config file.

    Args:
        configs (dict): The configuration to be updated.
        config_file_path (str): The path of the json config file.

    Returns:
        The updated configurations.

    """
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                loaded_config = json.load(f)
            dict_deep_update(configs, loaded_config)

        except json.JSONDecodeError:
            logging.exception('Decoding config file failed: {}'
                              .format(config_file_path))
    else:
        logging.warning('Update config skipped, no file found: {}'
                        .format(config_file_path))

    return configs


model_configs = {
    "0": {
        "human_detector": "cpu",
        "object_detector": "cpu",
        "face_detector": "none",
        "fire_detector": "none",
        "zawu_classifier": "none",
        "uniform_classifier": "none",
        "escalator_light_classifier": "none",
        "trashcan_classifier": "none",
        "escalator_maintenance_detector": "none",
        "falling_object_track_classifier": "none",
        "falling_object_type_classifier": "none",
        "cargo_detector": "none",
        "petroChina_detector": "none",
        "long_term_door_status_classifier": "none",
        "general_ocr_detector": "none",
        "general_ocr_recognitor": "none",
        "battery_bicycle_detector": "none",
        "lying_down_classifier": "none",
        "skeleton2D_detector": "none",
        "sleep_phone_detector": "none",
        "deep_sort": "none",
        "fall_slip_video_classifier": "none",
        "face_mask_detector": "none"
    },
    "1": {
        "human_detector": "0",
        "object_detector": "0",
        "face_detector": "none",
        "fire_detector": "none",
        "zawu_classifier": "none",
        "uniform_classifier": "none",
        "escalator_light_classifier": "none",
        "trashcan_classifier": "none",
        "escalator_maintenance_detector": "none",
        "falling_object_track_classifier": "none",
        "falling_object_type_classifier": "none",
        "cargo_detector": "none",
        "petroChina_detector": "none",
        "long_term_door_status_classifier": "none",
        "general_ocr_detector": "none",
        "general_ocr_recognitor": "none",
        "battery_bicycle_detector": "none",
        "lying_down_classifier": "none",
        "skeleton2D_detector": "none",
        "sleep_phone_detector": "none",
        "deep_sort": "none",
        "fall_slip_video_classifier": "none",
        "face_mask_detector": "none"
    },
    "2": {
        "human_detector": "0",
        "object_detector": "1",
        "face_detector": "none",
        "fire_detector": "none",
        "zawu_classifier": "none",
        "uniform_classifier": "none",
        "escalator_light_classifier": "none",
        "trashcan_classifier": "none",
        "escalator_maintenance_detector": "none",
        "falling_object_track_classifier": "none",
        "falling_object_type_classifier": "none",
        "cargo_detector": "none",
        "petroChina_detector": "none",
        "long_term_door_status_classifier": "none",
        "general_ocr_detector": "none",
        "general_ocr_recognitor": "none",
        "battery_bicycle_detector": "none",
        "lying_down_classifier": "none",
        "skeleton2D_detector": "none",
        "sleep_phone_detector": "none",
        "deep_sort": "none",
        "fall_slip_video_classifier": "none",
        "face_mask_detector": "none"
    },
    "3": {
        "human_detector": "0,1",
        "object_detector": "2",
        "face_detector": "none",
        "fire_detector": "none",
        "zawu_classifier": "none",
        "uniform_classifier": "none",
        "long_term_door_status_classifier": "none",
        "escalator_light_classifier": "none",
        "trashcan_classifier": "none",
        "escalator_maintenance_detector": "none",
        "falling_object_track_classifier": "none",
        "falling_object_type_classifier": "none",
        "cargo_detector": "none",
        "petroChina_detector": "none",
        "general_ocr_detector": "none",
        "general_ocr_recognitor": "none",
        "battery_bicycle_detector": "none",
        "lying_down_classifier": "none",
        "skeleton2D_detector": "none",
        "sleep_phone_detector": "none",
        "deep_sort": "none",
        "fall_slip_video_classifier": "none",
        "face_mask_detector": "none"
    },
    "4": {
        "human_detector": "0,1",
        "object_detector": "2,3",
        "face_detector": "none",
        "fire_detector": "none",
        "zawu_classifier": "none",
        "uniform_classifier": "none",
        "escalator_light_classifier": "none",
        "trashcan_classifier": "none",
        "escalator_maintenance_detector": "none",
        "falling_object_track_classifier": "none",
        "falling_object_type_classifier": "none",
        "cargo_detector": "none",
        "long_term_door_status_classifier": "none",
        "petroChina_detector": "none",
        "general_ocr_detector": "none",
        "general_ocr_recognitor": "none",
        "battery_bicycle_detector": "none",
        "lying_down_classifier": "none",
        "skeleton2D_detector": "none",
        "sleep_phone_detector": "none",
        "deep_sort": "none",
        "fall_slip_video_classifier": "none",
        "face_mask_detector": "none"
    },
    "version": {
        "human_detector": "v5",
        "object_detector": "v2",
        "face_detector": "v0",
        "fire_detector": "v1",
        "zawu_classifier": "v11",
        "uniform_classifier": "v1",
        "escalator_light_classifier": "v1",
        "trashcan_classifier": "v10",
        "long_term_door_status_classifier": "v1",
        "escalator_maintenance_detector": "v1",
        "falling_object_track_classifier": "v1",
        "falling_object_type_classifier": "v1",
        "battery_bicycle_detector": "v0",
        "cargo_detector": "v3",
        "petroChina_detector": "v0",
        "general_ocr_detector": "v0",
        "general_ocr_recognitor": "v1",
        "lying_down_classifier": "v2",
        "skeleton2D_detector": "v1",
        "sleep_phone_detector": "v1",
        "deep_sort": "v2",
        "fall_slip_video_classifier": "v2",
        "face_mask_detector": "v0"
    },
    "batch_size": {
        "human_detector": 1,
        "object_detector": 1,
        "face_detector": 1,
        "fire_detector": 1,
        "zawu_classifier": 1,
        "uniform_classifier": 1,
        "escalator_light_classifier": 1,
        "trashcan_classifier": 1,
        "escalator_maintenance_detector": 1,
        "falling_object_track_classifier": 1,
        "falling_object_type_classifier": 1,
        "cargo_detector": 1,
        "petroChina_detector": 1,
        "battery_bicycle_detector": 1,
        "long_term_door_status_classifier": 1,
        "general_ocr_detector": 1,
        "general_ocr_recognitor": 1,
        "lying_down_classifier": 1,
        "skeleton2D_detector": 1,
        "deep_sort": 1,
        "fall_slip_video_classifier": 1,
        "face_mask_detector": 1
    }
}

for gpu_num in range(5, 17):
    model_configs[str(gpu_num)] = {
        "human_detector": ",".join(map(str, range((gpu_num + 1) // 2))),
        "object_detector": ",".join(map(str, range((gpu_num + 1) // 2, gpu_num))),
        "face_detector": "none",
        "fire_detector": "none",
        "zawu_classifier": "none",
        "uniform_classifier": "none",
        "escalator_light_classifier": "none",
        "trashcan_classifier": "none",
        "escalator_maintenance_detector": "none",
        "falling_object_track_classifier": "none",
        "falling_object_type_classifier": "none",
        "cargo_detector": "none",
        "petroChina_detector": "none",
        "general_ocr_detector": "none",
        "general_ocr_recognitor": "none",
        "battery_bicycle_detector": "none",
        "lying_down_classifier": "none",
        "long_term_door_status_classifier": "none",
        "skeleton2D_detector": "none",
        "sleep_phone_detector": "none",
        "deep_sort": "none",
        "fall_slip_video_classifier": "none",
        "face_mask_detector": "none"
    }

# load model_config.json file if exist
model_config_file = "model_config.json"
model_configs = update_config(model_configs, model_config_file)


ModelSetting = namedtuple('ModelSetting', [
    'MODEL_NAME',
    'MODEL_VERSION',
    'NUM_CLASSES',
    'IMG_SHAPE',
    'CLIP_SHAPE',
    'MIN_TF_VERSION',
    'MODEL_PATH',
    'MODEL_HASH',

    # The following setting is used for models shipped with meta files.
    # If the archive path is not None, model synchronization will check
    # the archive file instead of the model file
    # Currently, only zip format is supported to archive model files.
    'MODEL_ARCHIVE_PATH',

    'LABEL_MAP_FILE',
    'INPUT_ELEMENT',
    'RETURN_ELEMENTS',
    'COLOR_SPACE',
])
default_model_setting = ModelSetting(
    MODEL_NAME=None,
    MODEL_VERSION='v0',
    NUM_CLASSES=None,
    IMG_SHAPE=None,
    CLIP_SHAPE=None,
    MIN_TF_VERSION='1.11.0',
    MODEL_PATH=None,
    MODEL_HASH=None,
    MODEL_ARCHIVE_PATH=None,
    LABEL_MAP_FILE=None,
    INPUT_ELEMENT=None,
    RETURN_ELEMENTS=None,
    COLOR_SPACE=None,
)


object_detector_settings = [
    default_model_setting._replace(
        # http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
        MODEL_NAME='object_detector',
        MODEL_VERSION='v0',
        NUM_CLASSES=90,
        MODEL_PATH='assets/d_graph.pb',
        MODEL_HASH='753c589fa596c855e42cf82eb463c637',
        LABEL_MAP_FILE='assets/label_map.pbtxt',
    ),
    default_model_setting._replace(
        # http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
        MODEL_NAME='object_detector',
        MODEL_VERSION='v1',
        NUM_CLASSES=90,
        MODEL_PATH='assets/d_graph_fcnn.pb',
        MODEL_HASH='1f1902262c16c2d9acb9bc4f8a8c266f',
        LABEL_MAP_FILE='assets/label_map.pbtxt',
    ),
    default_model_setting._replace(
        MODEL_NAME='object_detector',
        MODEL_VERSION='v2',
        NUM_CLASSES=90,
        IMG_SHAPE=(608, 608, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/yolov4_object_detector_v2/saved_model.pb',
        MODEL_HASH='bb47a9e27b6d76f91cd69d609ac2eb9d',
        MODEL_ARCHIVE_PATH='assets/yolov4_object_detector_v2.zip',
        LABEL_MAP_FILE='assets/label_map.pbtxt',
        COLOR_SPACE='rgb',
    ),
    # Support multi batch input
    default_model_setting._replace(
        MODEL_NAME='object_detector',
        MODEL_VERSION='v3',
        NUM_CLASSES=90,
        IMG_SHAPE=(608, 608, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/yolov4_object_detector_v3/saved_model.pb',
        MODEL_HASH='b7b49ac7c4487b9d00cf58ff8d6eb539',
        MODEL_ARCHIVE_PATH='assets/yolov4_object_detector_v3.zip',
        LABEL_MAP_FILE='assets/label_map.pbtxt',
        COLOR_SPACE='rgb',
    )
]


human_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='human_detector',
        MODEL_VERSION='v0',
        NUM_CLASSES=1,
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/retinanet_resnet50_retrain_model.pb',
        MODEL_HASH='ceaa5deb0a37ad0e643768d5c6f43bf3',
        LABEL_MAP_FILE='assets/label_map_human.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='human_detector',
        MODEL_VERSION='v1',
        NUM_CLASSES=1,
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/sagemaker-tf-object-detection-api-gpu-2019-12-06-17-09-04-193-61424.pb',
        MODEL_HASH='97fbc77e3e87bd929087cff10587e80b',
        LABEL_MAP_FILE='assets/label_map_human.pbtxt',
    ),
    default_model_setting._replace(
        MODEL_NAME='human_detector',
        MODEL_VERSION='v2',
        NUM_CLASSES=1,
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/tf-object-detection-api-gpu-ssd-resnet50-2020-10-16-12-27-24-654.pb',
        MODEL_HASH='973e88fd7b49cdc34232015a1f19b92f',
        LABEL_MAP_FILE='assets/label_map_human.pbtxt',
    ),
    default_model_setting._replace(
        MODEL_NAME='human_detector',
        MODEL_VERSION='v3',
        NUM_CLASSES=1,
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/yolov4_human_detector_v3/saved_model.pb',
        MODEL_HASH='dc143322659cc955250e4735bff6da25',
        MODEL_ARCHIVE_PATH='assets/yolov4_human_detector_v3.zip',
        LABEL_MAP_FILE='assets/label_map_human.pbtxt',
        COLOR_SPACE='rgb'
    ),
    default_model_setting._replace(
        MODEL_NAME='human_detector',  # the model support multi batch_size
        MODEL_VERSION='v4',
        NUM_CLASSES=1,
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/yolov4_human_detector_v4/saved_model.pb',
        MODEL_HASH='71f8bd863c9ae77e20e4783a19d1af8f',
        MODEL_ARCHIVE_PATH='assets/yolov4_human_detector_v4.zip',
        LABEL_MAP_FILE='assets/label_map_human.pbtxt',
    ),
    default_model_setting._replace(
        MODEL_NAME='human_detector',  # the model support multi batch_size
        MODEL_VERSION='v5',
        NUM_CLASSES=1,
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/yolov4_human_detector_v5/saved_model.pb',
        MODEL_HASH='0c2f2b5328c90891f75e34339109df42',
        MODEL_ARCHIVE_PATH='assets/yolov4_human_detector_v5.zip',
        LABEL_MAP_FILE='assets/label_map_human.pbtxt',
    )
]


sleep_phone_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='sleep_phone_detector',
        MODEL_VERSION='v0',
        IMG_SHAPE=(416, 416, 3),
        MIN_TF_VERSION='1.14.0',
        NUM_CLASSES=2,
        MODEL_PATH='assets/yolov4_sleep_play-phone_detector_v0/saved_model.pb',
        MODEL_HASH='7260b737a6c3e761606b2521b909d79e',
        MODEL_ARCHIVE_PATH='assets/yolov4_sleep_play-phone_detector_v0.zip',
        LABEL_MAP_FILE='assets/label_map_sleep_phone.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='sleep_phone_detector',
        MODEL_VERSION='v1',
        IMG_SHAPE=(416, 416, 3),
        MIN_TF_VERSION='1.14.0',
        NUM_CLASSES=2,
        MODEL_PATH='assets/yolov4_sleep_play-phone_detector_v1/saved_model.pb',
        MODEL_HASH='ac9a68daa1eacaa808e2e134a2f7e9ae',
        MODEL_ARCHIVE_PATH='assets/yolov4_sleep_play-phone_detector_v1.zip',
        LABEL_MAP_FILE='assets/label_map_sleep_phone.pbtxt'
    )
]


escalator_maintenance_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='escalator_maintenance_detector',
        MODEL_VERSION='v0',
        NUM_CLASSES=90,
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/escalator_maintenance_det_model_v0_ssd.pb',
        MODEL_HASH='10b8dd85b6a268c4a954fc2b69b6d141',
        LABEL_MAP_FILE='assets/label_map_escalator_maintenance.pbtxt',
        COLOR_SPACE='rgb',
    ),
    default_model_setting._replace(
        MODEL_NAME='escalator_maintenance_detector',
        MODEL_VERSION='v1',
        NUM_CLASSES=2,
        IMG_SHAPE=(608, 608, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/yolov4_em_detector_v1/saved_model.pb',
        MODEL_HASH='2512742fed0c50bcf80943c4d9553cd1',
        MODEL_ARCHIVE_PATH='assets/yolov4_em_detector_v1.zip',
        LABEL_MAP_FILE='assets/label_map_em_yoloV4.pbtxt',
        COLOR_SPACE='rgb',
    )
]


cargo_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='cargo_detector',
        MODEL_VERSION='v1',
        NUM_CLASSES=1,
        MIN_TF_VERSION='1.14.0',
        IMG_SHAPE=(416, 416, 3),
        MODEL_PATH='assets/yolov4-cargo-detector/saved_model.pb',
        MODEL_ARCHIVE_PATH='assets/yolov4-cargo-detector-v1.zip',
        MODEL_HASH='a3dedb1eea1e3ee6041bd414d65c6abf',
        LABEL_MAP_FILE='assets/label_map_cargo.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='cargo_detector',
        MODEL_VERSION='v2',
        NUM_CLASSES=4,
        MIN_TF_VERSION='1.14.0',
        IMG_SHAPE=(416, 416, 3),
        MODEL_PATH='assets/yolov4-cargo-detector-v2/saved_model.pb',
        MODEL_ARCHIVE_PATH='assets/yolov4-cargo-detector-v2.zip',
        MODEL_HASH='e7b4bc56f85038ab9673d14fbcca462d',
        LABEL_MAP_FILE='assets/label_map_cargo.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='cargo_detector',
        MODEL_VERSION='v3',
        NUM_CLASSES=4,
        MIN_TF_VERSION='1.14.0',
        IMG_SHAPE=(416, 416, 3),
        MODEL_PATH='assets/yolov4-cargo-detector-v3/saved_model.pb',
        MODEL_ARCHIVE_PATH='assets/yolov4-cargo-detector-v3.zip',
        MODEL_HASH='6f734b4a222dfaf4b8748314b9f3fa3b',
        LABEL_MAP_FILE='assets/label_map_cargo.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='cargo_detector',
        MODEL_VERSION='v4',
        NUM_CLASSES=2,
        MIN_TF_VERSION='1.14.0',
        IMG_SHAPE=(416, 416, 3),
        MODEL_PATH='assets/yolov4-cargo-detector-v4/saved_model.pb',
        MODEL_ARCHIVE_PATH='assets/yolov4-cargo-detector-v4.zip',
        MODEL_HASH='24c0889e4ba6052e1782450813160203',
        LABEL_MAP_FILE='assets/label_map_cargo_v2.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='cargo_detector',
        MODEL_VERSION='v5',
        NUM_CLASSES=2,
        MIN_TF_VERSION='1.14.0',
        IMG_SHAPE=(416, 416, 3),
        MODEL_PATH='assets/yolov4-cargo-detector-v5/saved_model.pb',
        MODEL_ARCHIVE_PATH='assets/yolov4-cargo-detector-v5.zip',
        MODEL_HASH='e42f45c1fe4c2b72ab815b25ee7a9f70',
        LABEL_MAP_FILE='assets/label_map_cargo_v2.pbtxt'
    )
]


face_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='face_detector',
        NUM_CLASSES=1,
        MODEL_PATH='assets/face_det_model.pb',
        MODEL_HASH='e50b08e5f0782ca3ba946b3ce5d2b356',
        LABEL_MAP_FILE='assets/label_map_face.pbtxt'
    ),
]


fire_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='fire_detector',
        MODEL_VERSION='v0',
        NUM_CLASSES=90,
        MODEL_PATH='assets/fire_det_model_v0_ssd.pb',
        MODEL_HASH='253cd2961eb5a60d9a32d5ba80f769da',
        LABEL_MAP_FILE='assets/label_map_fire.pbtxt',
        COLOR_SPACE='rgb'
    ),
    default_model_setting._replace(
        MODEL_NAME='fire_detector',
        MODEL_VERSION='v1',
        NUM_CLASSES=1,
        IMG_SHAPE=(608, 608, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/yolov4_fire_detector_v1/saved_model.pb',
        MODEL_HASH='ca9a64e7cfe1aaf144517ab271883afe',
        MODEL_ARCHIVE_PATH='assets/yolov4_fire_detector_v1.zip',
        LABEL_MAP_FILE='assets/label_map_fire_yoloV4.pbtxt',
        COLOR_SPACE='rgb',
    )
]


battery_bicycle_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='battery_bicycle_detector',
        MODEL_VERSION='v0',
        NUM_CLASSES=2,
        MIN_TF_VERSION='1.14.0',
        IMG_SHAPE=(416, 416, 3),
        MODEL_PATH='assets/yolov4-battery-bicycle-detector-v0/saved_model.pb',
        MODEL_ARCHIVE_PATH='assets/yolov4-battery-bicycle-detector-v0.zip',
        MODEL_HASH='f4a3805150eaf00bac077654f6bf744b',
        LABEL_MAP_FILE='assets/label_map_battery_bicycle.pbtxt'
    )
]


general_ocr_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='general_ocr_detector',
        MODEL_VERSION='v0',
        NUM_CLASSES=2,
        MODEL_PATH='assets/ocr_yolo_detector.pb',
        MODEL_HASH='4f454af480d1a112cf321d11c6e46478'
    )
]

general_ocr_recognitor_settings = [
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v0',
        MODEL_PATH='assets/ocr_and_alarm_recognition.pb',
        MODEL_HASH='0cc88bd8fd493a0e4e4c9afbf165a0a0'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v1',
        MODEL_PATH='assets/zsy_digit_recognition.pb',
        MODEL_HASH='256e10c256a3aa283d18ce636face9b2'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v2',
        MODEL_PATH='assets/dcd_lines_recognition.pb',
        MODEL_HASH='2049f84da9e37278756cf82d21820481'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v3',
        MODEL_PATH='assets/dcd_lines_recognition_1117.pb',
        MODEL_HASH='631c323f568bae2c802386ac5b0a7938'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v4',
        MODEL_PATH='assets/dcd_lines_recognition_1117_2.pb',
        MODEL_HASH='29acfcb640f4718489982ae8fdff1533'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v5',
        MODEL_PATH='assets/dcd_lines_recognition_1119.pb',
        MODEL_HASH='b9de353464f2ff379c5584fa43c20afe'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v6',
        MODEL_PATH='assets/dcd_lines_recognition_1120.pb',
        MODEL_HASH='279fca3880742f6c56016eaf6d0d5ed0'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v7',
        MODEL_PATH='assets/dcd_lines_recognition_1122.pb',
        MODEL_HASH='2ac7324cecd8428c1b02b25ddba40d91'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v8',
        MODEL_PATH='assets/dcd_lines_recognition_1123.pb',
        MODEL_HASH='5df2cf83ef008d7f064af60d25d43ac8'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v9',
        MODEL_PATH='assets/dcd_lines_recognition_1127.pb',
        MODEL_HASH='3db6fd90b014af909059143ed7bfa6ff'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v10',
        MODEL_PATH='assets/dcd_lines_recognition_1130.pb',
        MODEL_HASH='d21b5eb9f4f8949ba78814c9b39b1971'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v11',
        MODEL_PATH='assets/dcd_lines_recognition_1201.pb',
        MODEL_HASH='941a5a9e58271f45f5d62d7fe66de3ca'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v12',
        MODEL_PATH='assets/dcd_lines_recognition_1202.pb',
        MODEL_HASH='6bbf9ba23e3e092e34e00129ba9b19b4'
    ),
    default_model_setting._replace(
        MODEL_NAME='general_ocr_recognitor',
        MODEL_VERSION='v13',
        MODEL_PATH='assets/dcd_lines_recognition_1204.pb',
        MODEL_HASH='b6be0ce0e95cb04ac7886f394db3061e'
    )
]


petroChina_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='petroChina_detector',
        MODEL_VERSION='v0',
        NUM_CLASSES=90,
        MODEL_PATH='assets/petro_china_det_model_v0_ssd.pb',
        MODEL_HASH='75ce416829e13456e1a3fd850cd0cb55',
        LABEL_MAP_FILE='assets/label_map_petro_china.pbtxt',
        COLOR_SPACE='rgb'
    ),
]


face_mask_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='face_mask_detector',
        MODEL_VERSION='v0',
        IMG_SHAPE=(416, 416, 3),
        MIN_TF_VERSION='1.14.0',
        NUM_CLASSES=2,
        MODEL_PATH='assets/yolov4_face_mask_detector_v0/saved_model.pb',
        MODEL_HASH='4720f3e195ee3ecddff1ded7c87f04ab',
        MODEL_ARCHIVE_PATH='assets/yolov4_face_mask_detector_v0.zip',
        LABEL_MAP_FILE='assets/label_map_face_mask.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='face_mask_detector',
        MODEL_VERSION='v1',
        IMG_SHAPE=(416, 416, 3),
        MIN_TF_VERSION='1.14.0',
        NUM_CLASSES=2,
        MODEL_PATH='assets/yolov4_face_mask_detector_v1/saved_model.pb',
        MODEL_HASH='fc25eab09d98b90fc418e305f53fc9fb',
        MODEL_ARCHIVE_PATH='assets/yolov4_face_mask_detector_v1.zip',
        LABEL_MAP_FILE='assets/label_map_face_mask.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='face_mask_detector',
        MODEL_VERSION='v2',
        IMG_SHAPE=(416, 416, 3),
        MIN_TF_VERSION='1.14.0',
        NUM_CLASSES=2,
        MODEL_PATH='assets/yolov4_face_mask_detector_v2/saved_model.pb',
        MODEL_HASH='9bfdf0ed851b1a39a03ecef5d62da85d',
        MODEL_ARCHIVE_PATH='assets/yolov4_face_mask_detector_v2.zip',
        LABEL_MAP_FILE='assets/label_map_face_mask.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='face_mask_detector',
        MODEL_VERSION='v3',
        IMG_SHAPE=(608, 608, 3),
        MIN_TF_VERSION='1.14.0',
        NUM_CLASSES=2,
        MODEL_PATH='assets/yolov4_face_mask_detector_v3/saved_model.pb',
        MODEL_HASH='fe7fc2018d261b9d529341900a049310',
        MODEL_ARCHIVE_PATH='assets/yolov4_face_mask_detector_v3.zip',
        LABEL_MAP_FILE='assets/label_map_face_mask.pbtxt'
    ),
    default_model_setting._replace(
        MODEL_NAME='face_mask_detector',
        MODEL_VERSION='v4',
        IMG_SHAPE=(608, 608, 3),
        MIN_TF_VERSION='1.14.0',
        NUM_CLASSES=2,
        MODEL_PATH='assets/yolov4_face_mask_detector_v4/saved_model.pb',
        MODEL_HASH='6076d559aa05639b67c994c13e7861e3',
        MODEL_ARCHIVE_PATH='assets/yolov4_face_mask_detector_v4.zip',
        LABEL_MAP_FILE='assets/label_map_face_mask.pbtxt'
    )

]


fall_slip_video_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='fall_slip_video_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=2,
        CLIP_SHAPE=(56, 224, 224, 3),
        MODEL_PATH='assets/fall_slip_video_classiffier_v1.pb',
        MODEL_HASH='115d3f9293dd56e78ea8a8abdbe91a6f'
    ),
    default_model_setting._replace(
        MODEL_NAME='fall_slip_video_classifier',
        MODEL_VERSION='v2',
        NUM_CLASSES=2,
        CLIP_SHAPE=(56, 224, 224, 3),
        MODEL_PATH='assets/fall_slip_video_classifier_v2.pb',
        MODEL_HASH='cd600ee80b180381a60a674a34f09e70'
    )
]

zawu_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        NUM_CLASSES=2,
        IMG_SHAPE=(299, 299, 3),
        MODEL_PATH='assets/zawu_cls_model.pb',
        MODEL_HASH='afb19677eb99a88c36ac30e504648d85',
        INPUT_ELEMENT="ImageInputPlaceholder",
        RETURN_ELEMENTS=["Logits/final_tensor"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=4,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v1.pb',
        MODEL_HASH='7695033c86c94aa847a998a641b1d195',
        LABEL_MAP_FILE='assets/zawu_cls_model_v1.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v2',
        NUM_CLASSES=4,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v2.pb',
        MODEL_HASH='98d139794558c45ffd51becc50c0dcc9',
        LABEL_MAP_FILE='assets/zawu_cls_model_v2.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v4',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v4.pb',
        MODEL_HASH='def672ae41a84009362b5e1a4aeeedc6',
        LABEL_MAP_FILE='assets/zawu_cls_model_v4.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v5',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v5.pb',
        MODEL_HASH='ad4a019911528a1754bdf62f91a2f588',
        LABEL_MAP_FILE='assets/zawu_cls_model_v5.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v6',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v6.pb',
        MODEL_HASH='e428d6e0e88a20421bcad41d0a640dc5',
        LABEL_MAP_FILE='assets/zawu_cls_model_v6.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v7',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v7.pb',
        MODEL_HASH='4193aeb3f558967d3555f403dc9a9c0e',
        LABEL_MAP_FILE='assets/zawu_cls_model_v7.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v8',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v8.pb',
        MODEL_HASH='37627536014067b887e5280f7cd90c5b',
        LABEL_MAP_FILE='assets/zawu_cls_model_v8.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v10',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v10.pb',
        MODEL_HASH='59ac0a086207d36472e0a0fba539d8b6',
        LABEL_MAP_FILE='assets/zawu_cls_model_v10.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    ),
    default_model_setting._replace(
        MODEL_NAME='zawu_classifier',
        MODEL_VERSION='v11',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/zawu_cls_model_v11.pb',
        MODEL_HASH='3ec382126479759841299c2c91ca0aa3',
        LABEL_MAP_FILE='assets/zawu_cls_model_v11.txt',
        INPUT_ELEMENT="Placeholder",
        RETURN_ELEMENTS=["final_result"],
    )
]


uniform_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='uniform_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=2,
        IMG_SHAPE=(64, 64, 3),
        MODEL_PATH="assets/uniform_cls_model_v1.pb",
        MODEL_HASH="248db3ee4b13f73b12dc83bc1e300367",
    ),
    default_model_setting._replace(
        MODEL_NAME='uniform_classifier',
        MODEL_VERSION='v2',
        NUM_CLASSES=2,
        IMG_SHAPE=(299, 299, 3),
        MODEL_PATH="assets/uniform_cls_model_v2.pb",
        MODEL_HASH="5720a271ba75e9c6de98ad3f10c607d2",
    )
]


trashcan_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=2,
        IMG_SHAPE=(124, 124, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v1.pb',
        MODEL_HASH='d6b71474940e63f61b29afb9291a33fe',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v2',
        NUM_CLASSES=2,
        IMG_SHAPE=(124, 124, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v2.pb',
        MODEL_HASH='b4f85cf8dad9bbaac78a441424cc04e1',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v3',
        NUM_CLASSES=2,
        IMG_SHAPE=(124, 124, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v3.pb',
        MODEL_HASH='fcdf12623f4700e2a5dc64fc260cfa47',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v4',  # This model supports poly regions
        NUM_CLASSES=2,
        IMG_SHAPE=(128, 128, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v4.pb',
        MODEL_HASH='1974db6a834580f6e1f7aea8d420973b',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v5',
        NUM_CLASSES=2,
        IMG_SHAPE=(128, 128, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v5.pb',
        MODEL_HASH='456ebdd3e80439e7b72dd623f3f42962',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v6',
        NUM_CLASSES=2,
        IMG_SHAPE=(124, 124, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v6.pb',
        MODEL_HASH='d3fd3ae769319969efb5bfd084d22f3d',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v7',
        NUM_CLASSES=2,
        IMG_SHAPE=(124, 124, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v7.pb',
        MODEL_HASH='4b1969c35b3dea7d858239d9fbe91323',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v8',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v8.pb',
        MODEL_HASH='5bc74cd835c5b60d94f183ba75e78599',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v9',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v9.pb',
        MODEL_HASH='1a1a89edc7cf435d2595833eeb056308',
    ),
    default_model_setting._replace(
        MODEL_NAME='trashcan_classifier',
        MODEL_VERSION='v10',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/trashcan_cls_v10.pb',
        MODEL_HASH='3feab6651dac3f016fd0c66678f0f6de',
    ),
]


escalator_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='escalator_light_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=4,
        IMG_SHAPE=(64, 64, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/escalator_light_cls_model_v1.pb',
        MODEL_HASH='d409244aad71c13a7ab592fefb519889',
    ),
    default_model_setting._replace(
        MODEL_NAME='escalator_light_classifier',
        MODEL_VERSION='v2',
        NUM_CLASSES=4,
        IMG_SHAPE=(64, 64, 3),
        MODEL_PATH="assets/escalator_light_cls_model_v2.pb",
        MODEL_HASH="b9708624b10d9b1ed4ad1553fd387b5e",
    )
]


lying_down_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='lying_down_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=2,
        IMG_SHAPE=(3, 17, 17),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/lying_down_classifier_model_v1.pb',
        MODEL_HASH='b54e76e3d674d10339e02ce7fee71737',
    ),
    default_model_setting._replace(
        MODEL_NAME='lying_down_classifier',
        MODEL_VERSION='v2',
        NUM_CLASSES=2,
        IMG_SHAPE=(3, 17, 17),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/lying_down_classifier_model_v2.pb',
        MODEL_HASH='aba430565c793e17c6eb5ed3882f2724',
    )
]

skeleton2D_detector_settings = [
    default_model_setting._replace(
        MODEL_NAME='skeleton2D_detector',
        MODEL_VERSION='v1',
        IMG_SHAPE=(256, 192, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/skeleton2D_detector_model_v1.pb',
        MODEL_HASH='129515405ecba79574bc39139b0b1388',
    )
]


falling_objects_track_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='falling_object_track_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=2,
        IMG_SHAPE=(9, 1),
        MIN_TF_VERSION=None,
        MODEL_PATH='assets/falling_parameters.model',
        MODEL_HASH='acbbad8adf0728ac468d8b4752235ce1',
    )
]

falling_objects_type_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='falling_object_type_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=2,
        IMG_SHAPE=(32, 32, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/falling_cls_model.pb',
        MODEL_HASH='c501f9cc5fadd0e7177862b2acac0def',
    )
]


long_term_door_status_classifier_settings = [
    default_model_setting._replace(
        MODEL_NAME='long_term_door_status_classifier',
        MODEL_VERSION='v1',
        NUM_CLASSES=2,
        IMG_SHAPE=(224, 224, 3),
        MIN_TF_VERSION='1.14.0',
        MODEL_PATH='assets/long_term_door_status_model_v1.pb',
        MODEL_HASH='e8eb6a485cc5a553aef9376831afc048',
    )
]


deep_sort_tracker_settings = [
    default_model_setting._replace(
        MODEL_NAME='deep_sort',
        MODEL_VERSION='v3',
        MODEL_PATH='assets/deep_tracker/small-part-mars.pb',
        MODEL_HASH='5108dbe44d612a8163073cbb2514a168'
    ),
    default_model_setting._replace(
        MODEL_NAME='deep_sort',
        MODEL_VERSION='v2',
        MODEL_PATH='assets/deep_tracker/big-part-mars.pb',
        MODEL_HASH='5e4ceea3da0b751a5f0ca5e0021e0f31'
    ),
    default_model_setting._replace(
        MODEL_NAME='deep_sort',
        MODEL_VERSION='v1',
        MODEL_PATH='assets/deep_tracker/mars-big128.pb',
        MODEL_HASH='fbd51d64b6e2a6c5243c10e313219e58'
    ),
    default_model_setting._replace(
        MODEL_NAME='deep_sort',
        MODEL_VERSION='v0',
        MODEL_PATH='assets/deep_tracker/mars-small128.pb',
        MODEL_HASH='6849a8b571d654389713398dc6129d97'
    ),
]


def load_additional_model_settings(model_setting_file, existing_models=None):
    if not os.path.exists(model_setting_file):
        return []

    loaded_model_settings = []
    with open(model_setting_file, 'r') as f:
        try:
            raw_model_settings = json.load(f)
        except json.JSONDecodeError as e:
            logging.error("Failed to parse json: {}".format(e))
            return []

    if not existing_models:
        existing_models = []

    required_fields = ["MODEL_NAME", "MODEL_VERSION", "NUM_CLASSES",
                       "IMG_SHAPE", "MODEL_PATH", "MODEL_HASH"]

    for raw_ms in raw_model_settings:
        try:
            ms = default_model_setting._replace(**raw_ms)
        except ValueError as e:
            logging.error('Error converting model setting {}: {}'
                          .format(raw_ms, e))
            continue

        missing_required_fields = []
        for rf in required_fields:
            if not getattr(ms, rf):
                missing_required_fields.append(rf)

        if missing_required_fields:
            logging.error('Loaded model setting missing required field {}'
                          .format(missing_required_fields))
            continue

        if type(ms.IMG_SHAPE) != list:
            logging.error('The IMG_SHAPE must be a list or tuple, parsed "{}"'
                          .format(ms.IMG_SHAPE))
            continue

        ms = ms._replace(IMG_SHAPE=tuple(ms.IMG_SHAPE))

        version_conflict = False
        for old_ms in existing_models:
            if old_ms.MODEL_NAME == ms.MODEL_NAME and \
                    old_ms.MODEL_VERSION == ms.MODEL_VERSION:
                version_conflict = True
                break

        if version_conflict:
            logging.error('Model with the same name and version exists {}({})'
                          .format(ms.MODEL_NAME, ms.MODEL_VERSION))
            continue

        loaded_model_settings.append(ms)

    return loaded_model_settings


class ModelSettings:
    _models = object_detector_settings + \
        human_detector_settings + \
        face_detector_settings + \
        fire_detector_settings + \
        zawu_classifier_settings + \
        uniform_classifier_settings + \
        escalator_classifier_settings + \
        escalator_maintenance_detector_settings + \
        cargo_detector_settings + \
        falling_objects_track_classifier_settings + \
        falling_objects_type_classifier_settings + \
        petroChina_detector_settings + \
        general_ocr_detector_settings + \
        general_ocr_recognitor_settings + \
        long_term_door_status_classifier_settings + \
        battery_bicycle_detector_settings + \
        lying_down_classifier_settings + \
        skeleton2D_detector_settings + \
        sleep_phone_detector_settings + \
        deep_sort_tracker_settings + \
        fall_slip_video_classifier_settings + \
        face_mask_detector_settings + \
        trashcan_classifier_settings

    _model_setting_file = './additional_model_settings.json'

    @classmethod
    def initialize(cls):
        loaded_model_settings = load_additional_model_settings(
            cls._model_setting_file, cls._models)
        logging.info('loaded {} additional model settings: {}'
                     .format(len(loaded_model_settings), loaded_model_settings))
        cls._models += loaded_model_settings

        # fill default settings in model_configs
        all_model_names = set(m.MODEL_NAME for m in cls._models)
        for model_name in all_model_names:
            for key, _config in list(model_configs.items()):
                if model_name not in _config:
                    if key == 'version':
                        model_configs[key][model_name] = 'v0'
                    elif key == "batch_size":
                        model_configs[key][model_name] = 1
                    else:
                        model_configs[key][model_name] = 'none'

    @classmethod
    def get_version(cls, model_name: str):
        """Get the configured version of the given model.

        Returns:
            The version of the given model. Return None if the model is unknown.

        """
        return model_configs['version'].get(model_name, None)

    @classmethod
    def get_batch_size(cls, model_name: str):
        """Get the configured batch_size of the given model.

        Returns:
            The batch_size of the given model. Return 1 if the model is unknown.

        """
        return model_configs['batch_size'].get(model_name, 1)

    @classmethod
    def get_available_versions(cls, model_name: str):
        """Get all available versions of the given model.

        Returns:
            List of all available versions of the given model.

        """
        versions = []
        for ms in cls._models:
            if ms.MODEL_NAME == model_name:
                versions.append(ms.MODEL_VERSION)
        return versions

    @classmethod
    def get_model(cls, model_name: str, version=None):
        """Get model settings given the model name and version.

        Args:
            model_name (str): The model name.
            version (str): The model version.
                If None, use the version in model_configs.

        Returns:
            The model setting with the given name and version.
            None is model not found.

        """

        if not version:
            version = cls.get_version(model_name)

        if not version:
            logging.error('Failed to find model setting {}-{}'
                          .format(model_name, version))
            logging.info('The available model versions are: {}'
                         .format(cls.get_available_versions(model_name)))
            return None

        for ms in cls._models:
            if ms.MODEL_NAME == model_name and ms.MODEL_VERSION == version:
                return ms
        return None

    @classmethod
    def get_models(cls):
        """Get all registered model settings."""
        return cls._models.copy()


if __name__ == '__main__':
    model_settings = ModelSettings.get_models()
    name_path = []
    for model in model_settings:
        name_path_map = {
            'MODEL_NAME': model.MODEL_NAME,
            'MODEL_VERSION': model.MODEL_VERSION,
            'MODEL_PATH': model.MODEL_ARCHIVE_PATH or model.MODEL_PATH
        }
        name_path.append(name_path_map)
    with open('name_path_map.json', 'w') as f:
        json.dump(name_path, f, indent=4)
