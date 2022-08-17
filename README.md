# Detecting-Road-Damages-From-Image-And-Video
From training a yolov5 model for object recognition 
Part 01: Data Preprocessing
Folder Structure
![image](https://user-images.githubusercontent.com/97766408/185177855-ca78114a-9fc3-47d4-84ab-e69f49f4a140.png)
Now let us extract the data from the tar.gz file
Once the above piece of code executes, the extracted files will be stored in a folder called RoadDamageDataset inside the data folder. The project folder structure now looks as follows:
 ![image](https://user-images.githubusercontent.com/97766408/185178834-05c6c712-c8ae-4c93-a37e-9485e9d86740.png)

After extraction
If we peek into the RoadDamageDataset folder, we see the different locations at which the data was gathered.
 ![image](https://user-images.githubusercontent.com/97766408/185178886-e305eeb6-f566-454c-986f-b951203cceae.png)

Adachi, Chiba, Ichihara, etc are different cities in Japan where images of the road were taken and the defects were identified. Let us take a closer look.
For each city, the following folders are present
•	Annotations
•	ImageSets
•	JPEGImages
 ![image](https://user-images.githubusercontent.com/97766408/185179146-ebdf3719-6daf-4894-baf7-a235b7b1723b.png)

JPEGImages contain the color images of the roads
 ![image](https://user-images.githubusercontent.com/97766408/185179284-fa243421-13a5-4d82-851e-e0337fe3f341.png)

The folder “Annotations” contains the label details for each image
 ![image](https://user-images.githubusercontent.com/97766408/185179349-f2dc2b99-dd33-4de7-849f-621f60cd3791.png)

Image and annotations juxtaposed
For ease of understanding, the image and its corresponding annotations file are kept side-by-side. The markup of interest to us is <object> and the components inside it. For example, in this image, the defect present is D20 and it is located at the pixel point 87 to 226 on the X-axis and 281 to 432 on the Y-axis. Some images tend to contain more than one defect. The below image is an example. A pair of defects are actually visible to the naked eye.
 ![image](https://user-images.githubusercontent.com/97766408/185179447-8657b989-08c2-4006-8a57-43e9b50952fb.png)

Image with multiple defects on the road
The name of defects varies, generally starting with D, followed by a 2-digit number — like “D20”, “D01” etc.
Now that we have the annotations files and the images, we need to arrange them in such a way that they can be used to train a PyTorch model. Firstly, we need to put all images in one folder and put all the annotations in another. They should both have the same names, only their extensions will differ. For example, for a file “Chiba_20170913094022”, Chiba_20170913094022.jpg should be stored in one folder, and Chiba_20170913094022.xml should be stored in another folder. Let us take a look at the following code snippet that takes all the images and annotations for different cities and puts them all in two separate folders.
Output:
 ![image](https://user-images.githubusercontent.com/97766408/185179599-6cc64e0b-b6c7-4dbb-83f6-7047b1346c36.png)
The first part of the code (try block) creates the folders into which all the images and annotations will be kept. After the creation of the necessary folders, the arrangement looks like this:
 ![image](https://user-images.githubusercontent.com/97766408/185179820-3eb6f57e-61c4-43bd-acf3-da742aa59304.png)

Folders to store all images and annotations in one place
Once the second part of the code executes, the files will have been copied into the images and labels directory, depending on whether they were image files or XML ones.
 ![image](https://user-images.githubusercontent.com/97766408/185179905-56391b49-ee96-46a8-86b4-333cb6824448.png)

Files arranged to images and labels folder
The annotations file
Let us now take a closer look at the annotations files. We can pick any of them up at random.
 ![image](https://user-images.githubusercontent.com/97766408/185179989-9304f744-092e-4870-a35c-42e7fc34c75c.png)

This file contains coordinates of 5 defects
The above image is an example of a file containing 5 defects on the road. For each defect, we have the xmin, ymin, xmax and ymax — Basically the corners.
 
Annotations from the XML file
But we must convert this file to a format that is understandable by yolov5 when it trains. In the case of yolov5, it needs the x and y coordinates of the center of the highlighted section and the length and width of the same. Let us see what an ideal annotations file for yolov5 looks like. The same file shown above converted to yolov5 understandable format would look as follows.
 ![image](https://user-images.githubusercontent.com/97766408/185180084-67033c95-3940-4364-9f70-2987a4307261.png)

XML has become a txt file
Note how there are just 5 lines, one per defect. The first value is a numerical encoding of the class, followed by x_center, y_center, the width of the highlighted block, and the height of the highlighted block. Also, these are not absolute values, rather they are in proportion to the height or width of the image, as applicable. Let us try to understand this further.
The first value is the x co-ordinate of the center of the highlighted part divided by the length of the image. Let us take a look at the first highlighted section.
 ![image](https://user-images.githubusercontent.com/97766408/185180615-890730e9-4bb3-45b9-95c0-3fbb0e1ab5ab.png)
As mentioned in the original XML file for Adachi_20170906093900.xml
The xmin and xmax are 328 and 418 respectively. Therefore the x_center would be
(xmin + xmax)/2
Thus x_center is (328+418)/2 = 373
But that is not all. What yolov5 needs are a proportion value. What is the proportion of 373 with respect to the height of the image? If we look at the top section of the above image, we can see that height of the image is 600. Thus, proportion is 373/600 = 0.622
Similarly, y_center = (246 + 367) / 2 =306.5 and its corresponsing proportion value is 306.5 /600=0.511
Now for the width and height of the highlighted block.
width of highlighted block = (xmax — xmin)
But as we need a proportional value, it will be divided by the total width of the image.
width proportion = (xmax - xmin)/width of image
height proportion = (ymax - ymin)/height of image
For this example, width proportion = (418–328)/600 = 0.150
Height proportion = (367–246)/600 = 0.202
If we look at the first line of the transformed file, these are the values that we get.
 ![image](https://user-images.githubusercontent.com/97766408/185180742-254fdede-700e-40ec-a0f4-44a52c2acab7.png)
Yolov5 format for annotations
The good thing is that we already have some functions that would perform this transformation for us. Let us fix the goal first. Following is our directory structure
Current directory structure
![image](https://user-images.githubusercontent.com/97766408/185180813-045e64e5-a904-4151-98e2-84d09164e995.png)
What we want is another folder inside the assorted files directory, called annotations which will contain the annotations in the yolov5 understandable format.
 ![image](https://user-images.githubusercontent.com/97766408/185180996-181d3526-6c49-4718-aece-22c888e017ed.png)
What we want — annotations folder containing the same annotations as in the labels folder, but in a yolov5 understandable format
The highlighted section is what we want. Let us look at the code for that.
The first two functions, extract_info_from_xml() and convert_to_yolov5(…) are responsible for the extraction of data from XML files, followed by the conversion to proportion and consequent saving as txt files. After this step, the annotations folder will be programmatically created in the assorted files directory and the text files will be generated. Note here that we have created a mapping dictionary to convert the disaster types D11, D44, etc into numerics as the yolov5 only understands numbers.
Before we are done with the data pre-processing, let us quickly check if the annotations have been done properly. Let's run the following piece of code that picks up an image and its corresponding annotations txt file and draws bounding boxes according to the values present in the file.
We define a plot_bounding_box(…) function that draws a bounding box on images. Then we take up any old fille, visit its corresponding annotations (txt) file, and see if the coordinates in those files are actually highlighting defective regions of the road. The output looks as follows.
 ![image](https://user-images.githubusercontent.com/97766408/185181229-1129e8b8-3adc-4e0f-8775-2d22e6a45093.png)
            Defects clearly highlighted
The final step of data pre-processing is now to split the data into training, validation, and test. What we are going to have is the following.
 ![image](https://user-images.githubusercontent.com/97766408/185181339-4241b252-42d4-48ea-9b13-d17460125675.png)
Images and annotations split into training, validation, and test
The highlighted part in the above image shows the distribution. The images from assortedFiles/images will be re-distributed among finalRoad/images/train, finalRoad/images/val and finalRoad/images/test. Similarly, the annotations from assortedFiles/annotations will be distributed into finalRoad/annotations/train, finalRoad/annotations/val and finalRoad/annotations/test. Please remember that the files in the finalRoad/images/*/ should match the files in finalRoad/annotations/*/. That is for each file present in the train folder of images directory, it should have the corresponding annotation file in train folder of annotations directory. The following piece of code takes care of all that.
Code to mode images and annotations to train, Val and test
The first part of the code splits the file names into three lists for images and annotations, taking care that the same files are present in each group for images and annotations. Then we create the test, train, and vali folders. Finally, we copy the images from the assorted files folder to the final road folder.
 ![image](https://user-images.githubusercontent.com/97766408/185181449-3daf66f6-6926-4059-94e9-b0db48f85c94.png)
Populated the train, Val, and test folders.
Let us do a sanity check to see how many files are present in each folder
Output:
 ![image](https://user-images.githubusercontent.com/97766408/185181535-0c73c4c2-7ff6-411c-b022-2771dafa0dab.png)
Numbers add up. We can move to the next phase
All the code for part 1 can be found in the jupyter notebook here.
End of data pre-processing.
If you have come this far, take a dive into the next section where you will train a custom yolov5 model on the dataset. It may seem like an uphill journey but the rewards are satisfying.
You can open a new jupyter notebook now.
Part 02: Training
Get yolov5
This is where we download the code repository of yolov5 in order to take advantage of the already existing codebase.
 ![image](https://user-images.githubusercontent.com/97766408/185181632-c01188ff-5727-47a1-b805-22f8f1121b28.png)
Current directory structure
This is what the folder structure looks like. It is a continuation of what we have been doing in part 1. 01_Data_PreProcessing.ipynb was the notebook in which we wrote the code for data pre-processing (part 01). Now, we have a new notebook called 02_TrainYolo.ipynb where we will code part 02.
In the new jupyter notebook (02_TrainYolo.ipynb) that you just opened, execute the following:

Line 2 will download the yolov5 code repo into the current directory. Line 3 changes the working directory to inside the yolov5 folder which contains all the downloaded code repo. There is a requirement.txt file in the repository which contains all the necessary libraries. Line 4 installs them in the currently executing kernel.
 ![image](https://user-images.githubusercontent.com/97766408/185181933-f869ccfb-5594-4cef-8b6e-e68e4d62ae5c.png)
Successfully installed the libraries
If you check the folder structure now, you will see a folder called yolov5 in your codebase.
 ![image](https://user-images.githubusercontent.com/97766408/185181995-0b7a9a35-fb6c-49fb-8c96-2ee3254c1834.png)
We are inside the yolov5 folder now
The YAML file
This is the most crucial section of this part. We need to inform yolov5 regarding the whereabouts of the training, validation, and test files. This is communicated via a YAML file. The final YAML file looks like this, although we will create it programmatically.
 ![47](https://user-images.githubusercontent.com/97766408/185183091-a060e295-8894-4e84-b179-d79463c3164c.PNG)
The metadata for training
Note how we have given the absolute path to the train, validation, and test folder. This is important. And in order to reduce the confusion, we will use code to create the file. Following code snippet creates the YAML file — paste it in the jupyter notebook that you are currently working on and execute.
The first part of the code gets the absolute path to the parent directory as we are inside the yolov5 folder and the data is in a separate folder outside the yolov5 directory. The code results in creating a file called dataRoad.YAML inside the yolov5 folder.
 ![image](https://user-images.githubusercontent.com/97766408/185183209-67d9f1e3-bdbb-48c9-be4f-cc7be7193e6f.png)
Creating the metadata for training
Next, we will run the code to perform training. Actually, the code to train is already written in. yolov5/train.py. What we need to do is execute the code with the correct parameters. We do this as follows:
Execute train with the correct parameters
Rewriting the command here
! python train.py -— data dataRoad.yaml -— cfg yolov5s.yaml -— batch-size 32 -— epochs 1 -— name RoadTrainModel
An important parameter is --data where we give the name of the YAML file just created. Another necessary parameter is --epochs which are just 1 in the above example, but it should be changed to 20 or more to get better accuracy. Finally the parameter --name is important as it declares where the final trained model is going to be stored. For example, in this case, the trained model will be stored in the folder RoadTrainModel inside the run directory of the yolov5 folder.
 
Whenever we run the train.py file, a new RoadTrainModel* folder gets created.
The output of the above code is as follows:
 
Trained for 1 epoch
As you can see, I was training in a CPU system, so it took me 5 hours for one epoch! However, if you are using google colab, it should be much faster.
Trained weights
We need to be able to re-use the trained model. The weights of the model can be found in the RoadTrainModel* folder. * represents the highest number. For example, I had some errors and my final successful training happened on the 4th try — that is why in my case the trained weights will be found in RoadTrainModel4 directory.
 
                         best.pt is the prize that we are after
The best weights are stored in best.pt — this is the file that we need to load when we test our model.
All the code for part 02 can be found here.
You can now close this notebook and start a new one. Let us call it 03_Test.ipynb
Part 03: Testing
Once again we harness the code repository of yolov5 to test the trained network. It can be applied to images as well as videos, so we download a test image and a test video. Let us create a folder called test and save the image and video there. The folder structure looks like this:
 
Create a folder called test and add the image and video file
Also, note the 03_Test.ipynb file where we will code for this section. The code is pretty straightforward. We need to use the detect.py file inside the yolov5 directory and let it know which are the test files and where are the training weights. 

 
The output files have the same name as the input test files.
Notice the new folder called detect? That is where the outputs are going to be stored. The output of the code is as follows:
 ![image](https://user-images.githubusercontent.com/97766408/185183425-fff477e4-1c88-4425-9b98-0ea2d1196ca7.png)
Tells you where the output is stored
And the output file is as follows:
![image](https://user-images.githubusercontent.com/97766408/185183939-0f50fea0-4dd8-4c07-a777-857d3bbc9c4b.png)
                     Output image with detection
Thats it! You have been able to train and test a yolov5 object detection model on a custom dataset.
Last words
If you hung in there till the end, congratulations — I hope this time was well spent and you can come up with some great applications using the knowledge shared in this article. Feel free to comment or suggest improvements by getting in touch with me at gasmibouthaina1609@gmail.com Or 
                                  Gasmi.bouthainaa@gmail.com 

Detecting Road Damages From Image And Video

