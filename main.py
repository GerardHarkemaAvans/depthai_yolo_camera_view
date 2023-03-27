#!/usr/bin/env python3
# Import the library
import argparse
import os.path
import json


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
	data = json.load(f)
	f.close()
	print(data)
	print(data["nn_config"]["NN_specific_metadata"]["anchors"])
	
	

if __name__ == "__main__":
    main()
