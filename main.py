#!/usr/bin/env python3

# Import the library
import argparse
import os.path
import json

import sys
import cv2
import depthai as dai
import numpy as np
import time

def main():
	# Create the parser
	parser = argparse.ArgumentParser()
	# Add an argument
	parser.add_argument('-j', type=str, required=True)
	parser.add_argument('-b', type=str, required=True)
	# Parse the argument
	args = parser.parse_args()
	# Print "Hello" + the user input argument
	blob_filename = args.b
	json_filename = args.j
	#print('Hello,', blob_filename, json_filename)
	if(not os.path.isfile(blob_filename)):
		print('Error: File', blob_filename, 'does not exist')
		return
	if(not os.path.isfile(json_filename)):
		print('Error: File', json_filename, 'does not exist')
		return
	f = open(json_filename)
	jsonData = json.load(f)
	f.close()
	#print(jsonData)
	numClasses = jsonData["nn_config"]["NN_specific_metadata"]["classes"]
	print('numClasses:', numClasses)
	labelMap = jsonData["mappings"]["labels"]
	print('labelMap:', labelMap)
	anchors = jsonData["nn_config"]["NN_specific_metadata"]["anchors"]
	print('Anchors:', anchors)
	anchorMasks = jsonData["nn_config"]["NN_specific_metadata"]["anchor_masks"]
	print('anchorMasks:', anchorMasks)
	coordinateSize = jsonData["nn_config"]["NN_specific_metadata"]["coordinates"]
	print('coordinateSize:' , coordinateSize)
	iouThreshold = jsonData["nn_config"]["NN_specific_metadata"]["iou_threshold"]
	print('iouThreshold:', iouThreshold)
	confidenceThreshold = jsonData["nn_config"]["NN_specific_metadata"]["confidence_threshold"]
	print('confidenceThreshold:', confidenceThreshold)
	inputSize = jsonData["nn_config"]["input_size"]
	print("inputSize:", inputSize)
	inputSizeX, inputSizeY = inputSize.split("x")
	inputSizeX  = int(inputSizeX)
	inputSizeY  = int(inputSizeY)

	syncNN = True

	# Create pipeline
	pipeline = dai.Pipeline()

	# Define sources and outputs
	camRgb = pipeline.create(dai.node.ColorCamera)
	spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
	monoLeft = pipeline.create(dai.node.MonoCamera)
	monoRight = pipeline.create(dai.node.MonoCamera)
	stereo = pipeline.create(dai.node.StereoDepth)
	nnNetworkOut = pipeline.create(dai.node.XLinkOut)

	xoutRgb = pipeline.create(dai.node.XLinkOut)
	xoutNN = pipeline.create(dai.node.XLinkOut)
	xoutDepth = pipeline.create(dai.node.XLinkOut)

	xoutRgb.setStreamName("rgb")
	xoutNN.setStreamName("detections")
	xoutDepth.setStreamName("depth")
	nnNetworkOut.setStreamName("nnNetwork")

	# Properties
	camRgb.setPreviewSize(inputSizeX, inputSizeY)
	camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
	camRgb.setInterleaved(False)
	camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

	monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
	monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

	# setting node configs
	stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
	# Align depth map to the perspective of RGB camera, on which inference is done
	stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
	stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

	spatialDetectionNetwork.setBlobPath(blob_filename)
	spatialDetectionNetwork.setConfidenceThreshold(confidenceThreshold)
	spatialDetectionNetwork.input.setBlocking(False)
	spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
	spatialDetectionNetwork.setDepthLowerThreshold(100)
	spatialDetectionNetwork.setDepthUpperThreshold(5000)

	# Yolo specific parameters
	spatialDetectionNetwork.setNumClasses(numClasses)
	spatialDetectionNetwork.setCoordinateSize(coordinateSize)
	spatialDetectionNetwork.setAnchors(anchors)
	spatialDetectionNetwork.setAnchorMasks(anchorMasks)
	spatialDetectionNetwork.setIouThreshold(iouThreshold)

	# Linking
	monoLeft.out.link(stereo.left)
	monoRight.out.link(stereo.right)

	camRgb.preview.link(spatialDetectionNetwork.input)
	if syncNN:
	    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
	else:
	    camRgb.preview.link(xoutRgb.input)

	spatialDetectionNetwork.out.link(xoutNN.input)

	stereo.depth.link(spatialDetectionNetwork.inputDepth)
	spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
	spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

	# Connect to device and start pipeline
	with dai.Device(pipeline) as device:

		# Output queues will be used to get the rgb frames and nn jsonData from the outputs defined above
		previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
		detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
		depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
		networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

		startTime = time.monotonic()
		counter = 0
		fps = 0
		color = (255, 255, 255)
		printOutputLayersOnce = True

		while True:
			inPreview = previewQueue.get()
			inDet = detectionNNQueue.get()
			depth = depthQueue.get()
			inNN = networkQueue.get()

			if printOutputLayersOnce:
				toPrint = 'Output layer names:'
				for ten in inNN.getAllLayerNames():
				    toPrint = f'{toPrint} {ten},'
				print(toPrint)
				printOutputLayersOnce = False;

			frame = inPreview.getCvFrame()
			depthFrame = depth.getFrame() # depthFrame values are in millimeters

			depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
			depthFrameColor = cv2.equalizeHist(depthFrameColor)
			depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

			counter+=1
			current_time = time.monotonic()
			if (current_time - startTime) > 1 :
				fps = counter / (current_time - startTime)
				counter = 0
				startTime = current_time

			detections = inDet.detections

			# If the frame is available, draw bounding boxes on it and show the frame
			height = frame.shape[0]
			width  = frame.shape[1]
			for detection in detections:
				roijsonData = detection.boundingBoxMapping
				roi = roijsonData.roi
				roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
				topLeft = roi.topLeft()
				bottomRight = roi.bottomRight()
				xmin = int(topLeft.x)
				ymin = int(topLeft.y)
				xmax = int(bottomRight.x)
				ymax = int(bottomRight.y)
				cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

				# Denormalize bounding box
				x1 = int(detection.xmin * width)
				x2 = int(detection.xmax * width)
				y1 = int(detection.ymin * height)
				y2 = int(detection.ymax * height)
				try:
				    label = labelMap[detection.label]
				except:
				    label = detection.label
				cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
				cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
				cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
				cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
				cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

				cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

			cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
			cv2.imshow("depth", depthFrameColor)
			cv2.imshow("rgb", frame)

			if cv2.waitKey(1) == ord('q'):
				break

if __name__ == "__main__":
    main()
