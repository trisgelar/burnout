import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# print("TF version:", tf.__version__)
# if tf.config.list_physical_devices('GPU'):
#     print("GPU is available",'\n')
#     print(tf.config.list_physical_devices('GPU'))
# else:
#     print("GPU is not available", '\n')

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The export expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.8,
                                           tf.uint8)
        print(detection_masks_reframed)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def main():
    st.header("Realtime Burnout Detection Application")

    burnout_detection_page = "Video Based burnout detection (sendrecv)"
    realtime_burnout_detection_page = "Microscope Based burnout detection (sendrecv)"
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            burnout_detection_page,
            realtime_burnout_detection_page,
        ],
    )
    st.subheader(app_mode)

    if app_mode == burnout_detection_page:
        app_burnout_detection()
    elif app_mode == realtime_burnout_detection_page:
        app_realtime_burnout_detection()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_realtime_burnout_detection():
    PATH_TO_LABELS = os.path.join('models', 'labelmap.pbtxt')
    PATH_TO_SAVED_MODEL = os.path.join('models', 'saved_model')

    class OpenCVVideoProcessor(VideoProcessorBase):
        type: Literal["noop", "burnout"]

        def __init__(self) -> None:
            self.type = "burnout"
            self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
            self.detection_model = tf.saved_model.load(str(PATH_TO_SAVED_MODEL))


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "burnout":
                output_dict = run_inference_for_single_image(self.detection_model, img)
                img = vis_util.visualize_boxes_and_labels_on_image_array( img, 
                    output_dict['detection_boxes'], 
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    self.category_index,
                    instance_masks=output_dict.get('detection_masks_reframed', None),
                    use_normalized_coordinates=True,
                    min_score_thresh=.5,
                    line_thickness=8
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key=f"realtime-burnout-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.type = st.radio(
            "Select transform type", ("burnout", "noop")
        )

def app_burnout_detection():
    """ Media streamings """
    MEDIAFILES = {
        "test.mp4 (local)": {
            "local_file_path": HERE / "data/test.mp4",
            "type": "video",
        },
    }
    media_file_label = "test.mp4 (local)"
    media_file_info = MEDIAFILES[media_file_label]

    PATH_TO_LABELS = os.path.join('models', 'labelmap.pbtxt')
    PATH_TO_SAVED_MODEL = os.path.join('models', 'saved_model')

    def create_player():
        if "local_file_path" in media_file_info:
            return MediaPlayer(str(media_file_info["local_file_path"]))

        # NOTE: To stream the video from webcam, use the code below.
        # return MediaPlayer(
        #     "1:none",
        #     format="avfoundation",
        #     options={"framerate": "30", "video_size": "1280x720"},
        # )

    class OpenCVVideoProcessor(VideoProcessorBase):
        type: Literal["noop", "burnout"]

        def __init__(self) -> None:
            self.type = "burnout"
            self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
            self.detection_model = tf.saved_model.load(str(PATH_TO_SAVED_MODEL))


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "burnout":
                output_dict = run_inference_for_single_image(self.detection_model, img)
                img = vis_util.visualize_boxes_and_labels_on_image_array( img, 
                    output_dict['detection_boxes'], 
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    self.category_index,
                    instance_masks=output_dict.get('detection_masks_reframed', None),
                    use_normalized_coordinates=True,
                    min_score_thresh=.5,
                    line_thickness=8
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    WEBRTC_CLIENT_SETTINGS.update(
        {
            "media_stream_constraints": {
                "video": media_file_info["type"] == "video",
                "audio": media_file_info["type"] == "audio",
            }
        }
    )

    webrtc_ctx = webrtc_streamer(
        key=f"media-streaming-{media_file_label}",
        mode=WebRtcMode.RECVONLY,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        player_factory=create_player,
        video_processor_factory=OpenCVVideoProcessor,
    )

    if media_file_info["type"] == "video" and webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.type = st.radio(
            "Select transform type", ("burnout", "noop")
        )


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()