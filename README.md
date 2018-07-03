# Plant-Detection-Using-TensorFlow

__Plant identification based on leaf structure__

## Introduction 

  Plants exist everywhere we live, as well as places without us. Many of them carry significant information for the development of human society. The relationship between human beings and plants are also very close. In addition, plants are important means of circumstances and production of human beings. Regrettably, the amazing development of human civilization has disturbed this balance to a greater extent than realized. It is one of the biggest duties of human beings to save the plants from various dangers. So, the diverseness of the plant community should be restored and put everything back to balance. The urgent situation is that many plants are at the risk of extinction. So, it is very necessary to set up a database for plant protection We believe that the first step is to teach a computer how to classify plants. 
  
  The tutorial is written for Windows 10, and it will also work for Windows 7 and 8. The general procedure can also be used for Linux operating systems, but file paths and package installation commands will need to change accordingly.

__Special Thanks To: EdjeElectronics, Sentdex__

If you encounter any problems while doing this project please do refer the link given below for the solutions https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 

## Steps

### 1. Install TensorFlow (skip this step if TensorFlow-GPU 1.5 is already installed) or TensorFlow-CPU

Install TensorFlow-GPU and CPU by following the instructions or you can follow YouTube Video by Mark Jay.

  The video is made for TensorFlow-GPU v1.4, but the “pip install --upgrade tensorflow-gpu or pip install --upgrade tensorflow (FOR CPU)” command will automatically download version 1.5. Download and install CUDA v9.0 and cuDNN v7.0 (rather than CUDA v8.0 and cuDNN v6.0 as instructed in the video), because they are supported by TensorFlow-GPU v1.5. As future versions of TensorFlow are released, you will likely need to continue updating the CUDA and cuDNN versions to the latest supported version.
  Be sure to install Anaconda with Python 3.6 as instructed in the video, as the Anaconda virtual environment will be used for the rest of this tutorial.
  Visit TensorFlow's website for further installation details, including how to install it on other operating systems (like Linux). The object detection repository itself also has installation instructions.

### 2. Set up TensorFlow Directory and Anaconda Virtual Environment

  The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or train an object detection model.
This portion of the tutorial goes over the full set up required. It is fairly meticulous, but follow the instructions closely, because improper setup can cause unwieldy errors down the road.

#### 2a. Download TensorFlow Object Detection API repository from GitHub

Create a folder directly in C: and name it “tensorflow1”. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.
Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into the C:\tensorflow1 directory you just created. Rename “models-master” to just “models”. (Note, this tutorial was done using this GitHub commit of the TensorFlow Object Detection API. If portions of this tutorial do not work, it may be necessary to download and use this exact commit rather than the most up-to-date version.)

#### 2b. Download the ssd_mobilenet_v1_coco model from TensorFlow's model zoo
(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
	TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo. Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy. I initially started with the SSD-MobileNet-V1 model because my local machine(laptop) configurations is lower and I am training my dataset on CPU (no GPU). If you have the higher configuration laptop with decent NVDIA graphics card then you can make use of Faster-RCNN-Inception-V2 model, and the detection works considerably better, but with a noticeably slower speed. This tutorial will use the ssd_mobilenet_v1_coco model. Download the model here (http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz). Open the downloaded ssd_mobilenet_v1_coco file with a file archiver such as WinZip or 7-Zip and extract the ssd_mobilenet_v1_coco folder to the 
C:\tensorflow1\models\research\object_detection folder. (Note: The model date and version will likely change in the future, but it should still work with this tutorial.)

#### 2c. Download this tutorial's repository from GitHub

Download the full repository located on this page (scroll to the top and click Clone or Download) and extract all the contents directly into the C:\tensorflow1\models\research\object_detection directory. (You can overwrite the existing "README.md" file.) This establishes a specific directory structure that will be used for the rest of the tutorial.

link = (https://github.com/KundanBalse/Plant-Detection-Using-TensorFlow)

Delete the following files (do not delete the folders):

•	All files in \object_detection\images\train and \object_detection\images\test

•	The “test_labels.csv” and “train_labels.csv” files in \object_detection\images

•	All files in \object_detection\training

•	All files in \object_detection\inference_graph

Now, you are ready to start from scratch in training your own Plant detector. This tutorial will assume that all the files listed above were deleted and will go on to explain how to generate the files for your own training dataset.

#### 2d. Set up new Anaconda virtual environment

Next, we'll work on setting up a virtual environment in Anaconda for tensorflow. From the Start menu in Windows, search for the Anaconda Prompt utility, right click on it, and click “Run as Administrator”. If Windows asks you if you would like to allow it to make changes to your computer, click Yes.
In the command terminal that pops up, create a new virtual environment called “tensorflow1” by issuing the following command:

C:\> conda create -n tensorflow1 pip python=3.5

Then, activate the environment by issuing:

C:\> activate tensorflow1

Install tensorflow in this environment by issuing:

(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow

Install the other necessary packages by issuing the following commands:

(tensorflow1) C:\> conda install -c anaconda protobuf

(tensorflow1) C:\> pip install pillow

(tensorflow1) C:\> pip install lxml

(tensorflow1) C:\> pip install Cython

(tensorflow1) C:\> pip install jupyter

(tensorflow1) C:\> pip install matplotlib

(tensorflow1) C:\> pip install pandas

(tensorflow1) C:\> pip install opencv-python

(Note: The ‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow, but they are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.)

#### 2e. Configure PYTHONPATH environment variable

A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. Do this by issuing the following commands (from any directory):

	(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim

(Note: Every time the "tensorflow1" virtual environment is exited, the PYTHONPATH variable is reset and needs to be set up again.)


#### 2f. Compile Protobufs and run setup.py

Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters. Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API installation page does not work on Windows. Every .proto file in the \object_detection\protos directory must be called out individually by the command.
In the Anaconda Command Prompt, change directories to the \models\research directory and copy and paste the following command into the command line and press Enter: 

	(protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto)

This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.
(Note: TensorFlow occassionally adds new .proto files to the \protos folder. If you get an error saying ImportError: cannot import name 'something_something_pb2', you may need to update the protoc command to include the new .proto files.)
Finally, run the following commands from the C:\tensorflow1\models\research directory:

	(tensorflow1) C:\tensorflow1\models\research> python setup.py build

	(tensorflow1) C:\tensorflow1\models\research> python setup.py install

#### 2g. Test TensorFlow setup to verify it works

  The TensorFlow Object Detection API is now all set up to use pre-trained models for object detection, or to train a new one. You can test it out and verify your installation is working by launching the object_detection_tutorial.ipynb script with Jupyter. From the \object_detection directory, issue this command:
  
	(tensorflow1)C:\tensorflow1\models\research\object_detection>juprter notebook object_detection_tutorial.ipynb

  This opens the script in your default web browser and allows you to step through the code one section at a time. You can step through each section by clicking the “Run” button in the upper toolbar. The section is done running when the “In [* ]” text next to the section populates with a number.
(Note: part of the script downloads the ssd_mobilenet_v1 model from GitHub, which is about 74MB. This means it will take some time to complete the section, so be patient.)
  Once you have stepped all the way through the script, you should see two labelled images at the bottom section the page. If you see this, then everything is working properly! If not, the bottom section will report any errors encountered. See the Appendix for a list of errors I encountered while setting this up.
  
  

### 3. Gather and Label Images

  Now that the TensorFlow Object Detection API is all set up and ready to go, we need to provide the images it will use to train a new detection classifier.
 

#### 3a. Collect Images

  TensorFlow needs hundreds of images of an object to train a good detection classifier. To train a robust classifier, the training images should have random plants in the image along with the desired plants and should have a variety of backgrounds and lighting conditions. There should be some images where the desired plant is partially obscured, overlapped with something else, or only halfway in the picture.
  
   <img src = "https://github.com/KundanBalse/Plant-Detection-Using-TensorFlow/blob/master/Documents/Documents.jpg">
   
  For my plant Detection classifier, I have 5 different plants I want to detect (ivy tree, garden geranium, common guava, sago cycad, painters palette). I used my cell phone (Redmi note 4) to take about 80 pictures of each plant on its own, with various other non-desired objects in the pictures. And also, some images with overlapped leaves so that I can detect the plants effectively. Totally I took around 480 images of 5 different plants each having approx. 80 images.
Make sure the images aren’t too large. They should be less than 200KB each, and their resolution shouldn’t be more than 720x1280. The larger the images are, the longer it will take to train the classifier. You can use the resizer.py script in this repository to reduce the size of the images.

  After you have all the pictures you need, move 20% of them to the \object_detection\images\test directory, and 80% of them to the \object_detection\images\train directory. Make sure there are a variety of pictures in both the \test and \train directories.

#### 3b. Label Images

  Here comes the fun part! With all the pictures gathered, it’s time to label the desired objects in every picture. LabelImg is a great tool for labeling images, and its GitHub page has very clear instructions on how to install and use it.
  
LabelImg download link (https://tzutalin.github.io/labelImg/)

Download and install LabelImg, point it to your \images\train directory, and then draw a box around each plant leaf in each image. Repeat the process for all the images in the \images\test directory. This will take a while!
LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.
<img src="https://github.com/KundanBalse/Plant-Detection-Using-TensorFlow/blob/master/Documents/tt.png">

### 4. Generate Training Data

  First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:
  
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py

This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder.

Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file in Step 5b.
For example, say you are training a classifier to detect basketballs, shirts, and shoes. You will replace the following code in generate_record.py:

#To-do this replace with labelmap
~~~
def class_text_to_int(row_label):
    if row_label == 'common guava':
        return 1
    elif row_label == 'ivy tree':
        return 2
    elif row_label == 'garden geranium':
        return 3
    elif row_label == 'painters palette':
        return 4
    elif row_label == 'sago cycad':
        return 5
    else:
        None
~~~
        
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:

	(python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record)

	(python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record)

These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

### 5. Create Label Map and Configure Training

  The last thing to do before training is to create a label map and edit the training configuration file.

#### 5a. Label map

  The label map tells the trainer what each plant is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. (Make sure the file type is. pbtxt, not .txt!) In the text editor, copy or type in the label map in the format below (the example below is the label map for my Plant Detector):
  
~~~
  
item {
  id: 1
  name: 'common guava'
}
item {
  id: 2
  name: 'ivy tree'
}
item {
  id: 3
  name: 'garden geranium'
}
item {
  id: 4
  name: 'painters palette'
}
item {
  id: 5
  name: 'sago cycad'
}

~~~
The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file.

#### 5b. Configure training

  Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!
Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the ssd_mobilenet_v1_pets.config file into the \object_detection\training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

  Make the following changes to the faster_rcnn_inception_v2_pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).
Line 9. Change num_classes to the number of different objects you want the classifier to detect it would be num_classes : 5 (because 5 different plants)

Line 110. Change fine_tune_checkpoint to:
fine_tune_checkpoint:"C:/tensorflow1/models/research/object_detection ssd_mobilenet_v1_coco_2017_11_17 /model.ckpt"

Lines 126 and 128. In the train_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/train.record"

label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

Line 132. Change num_examples to the number of images you have in the \images\test directory.

Lines 140 and 142. In the eval_input_reader section, change input_path and label_map_path to:

input_path: "C:/tensorflow1/models/research/object_detection/test.record"

label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

### 6. Run the Training

Here we go! From the \object_detection directory, issue the following command to begin training:

	(python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ ssd_mobilenet_v1_pets.config)

If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins.

Each step of training reports the loss. It will start high and get lower and lower as training progresses. For my training on the Faster-RCNN-Inception-V2 model, it started at about 3.0 and quickly dropped below 0.8. I recommend allowing your model to train until the loss consistently drops below 0.05, which will take about 40,000 steps, or about 2 hours (depending on how powerful your CPU and GPU are). Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 20 and should be trained until the loss is consistently under 2.
You can view the progress of the training job by using TensorBoard. To do this, open a new instance of Anaconda Prompt, activate the tensorflow1 virtual environment, change to the C:\tensorflow1\models\research\object_detection directory, and issue the following command:

	(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training

This will create a webpage on your local machine at YourPCName:6006, which can be viewed through a web browser. The TensorBoard page provides information and graphs that show how the training is progressing. One important graph is the Loss graph, which shows the overall loss of the classifier over time.
The training routine periodically saves checkpoints about every five minutes. You can terminate the training by pressing Ctrl+C while in the command prompt window. I typically wait until just after a checkpoint has been saved to terminate the training. You can terminate training and start it later, and it will restart from the last saved checkpoint. The checkpoint at the highest number of steps will be used to generate the frozen inference graph.
<img src="https://github.com/KundanBalse/Plant-Detection-Using-TensorFlow/blob/master/Documents/ddd.png">

### 7. Export Inference Graph

  Now that training is complete, the last step is to generate the frozen inference graph (. pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

	(python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph)

This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The .pb file contains the object detection classifier.
<img src = "https://github.com/KundanBalse/Plant-Detection-Using-TensorFlow/blob/master/Documents/dd.png">

### 8. Use Your Newly Trained Object Detection Classifier!

  The Plant detector is all ready to go! I’ve written Python scripts to test it out on an image, video, or webcam feed.
Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes you want to detect. (For my Plant Detector, there are 5 plants I want to detect, so NUM_CLASSES = 5.)

To test your object detector, move a picture of the object or objects into the \object_detection folder, and change the IMAGE_NAME variable in the Object_detection_image.py to match the file name of the picture. Alternatively, you can use a video of the objects (using Object_detection_video.py), or just plug in a USB webcam and point it at the objects (using Object_detection_webcam.py).

To run any of the scripts, type “idle” in the Anaconda Command Prompt (with the “tensorflow1” virtual environment activated) and press ENTER. This will open IDLE, and from there, you can open any of the scripts and run them.

If everything is working properly, the object detector will initialize for about 10 seconds and then display a window showing any objects it’s detected in the image!

If you encounter errors, please check out the Appendix: it has a list of errors that I ran in to while setting up my object detection classifier. You can also trying Googling the error. There is usually useful information on Stack Exchange or in TensorFlow’s Issues on GitHub.

__link for Plant Identification in Real Time Video__ = (https://drive.google.com/open?id=1nc7SAEPdD5AvG17GJfKLj-O80X1QlmZO)

<img src = "https://github.com/KundanBalse/Plant-Detection-Using-TensorFlow/blob/master/Documents/er.jpg">

### Appendix: Common Errors

It appears that the TensorFlow Object Detection API was developed on a Linux-based operating system, and most of the directions given by the documentation are for a Linux OS. Trying to get a Linux-developed software library to work on Windows can be challenging. There are many little snags that I ran in to while trying to set up tensorflow-gpu to train an object detection classifier on Windows 10. This Appendix is a list of errors I ran in to, and their resolutions.

__1. ModuleNotFoundError: No module named 'deployment'__

This error occurs when you try to run object_detection_tutorial.ipynb or train.py and you don’t have the PATH and PYTHONPATH environment variables set up correctly. Exit the virtual environment
by closing and re-opening the Anaconda Prompt window. Then, issue “activate tensorflow1” to re-enter the environment, and then issue the commands given in Step 2e.
You can use “echo %PATH%” and “echo %PYTHONPATH%” to check the environment variables and make sure they are set up correctly.
Also, make sure you have run these commands from the \models\research directory:

setup.py build
setup.py install

__2. ImportError: cannot import name 'preprocessor_pb2'__

ImportError: cannot import name 'string_int_label_map_pb2'
(or similar errors with other pb2 files)
This occurs when the protobuf files (in this case, preprocessor.proto) have not been compiled. Re-run the protoc command given in Step 2f. Check the \object_detection\protos folder to make sure there is a name_pb2.py file for every name.proto file.

__3. object_detection/protos/.proto: No such file or directory__

This occurs when you try to run the
“protoc object_detection/protos/*.proto --python_out=.”
command given on the TensorFlow Object Detection API installation page. Sorry, it doesn’t work on Windows! Copy and paste the full command given in Step 2f instead. There’s probably a more graceful way to do it, but I don’t know what it is.

__4. Unsuccessful TensorSliceReader constructor:Failed to get "file path" … The filename, directory name, or volume label syntax is incorrect.__

This error occurs when the filepaths in the training configuration file (faster_rcnn_inception_v2_pets.config or similar) have not been entered with backslashes instead of forward slashes. Open the .config file and make sure all file paths are given in the following format: “C:/path/to/model.file”

