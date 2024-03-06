This project is created for PDF-scans handwritten Russian character recognition.

# Repository structure:
- ./models # Includes pretrained and fine-tuned Paddle-based OCR model
- ./src # Includes utility code responsible for PDF data extraction, preprocessing and recognition
- ./scripts # Includes .sh scripts for model training and inference
- ./configs # Includes .yaml model configuration, default paths PATHS.txt file
- ./outputs # Includes original, preprocessed and cropped images and a csv file containing recognition results
  
# Model inference:
Firsly, confirm that you have edited default SCANS_FOLDER_PATH from PATHS.txt file in the ./configs folder as
it now should point to your scanns path directory.

For use of default pretrained and fine-tuned model call ./scripts/inference.sh which will automatically 
fetch latest file from the SCANS_FOLDER_PATH config parameter and place all the outputs in the output/folder

# Custom model inference: 
Firsly, confirm that you have edited default CHARACTER_DICT_PATH, INFERENCE_MODEL_PATH from PATHS.txt file in the ./configs folder
as they now should point to your models dictionary and your models parameters accordingly.

For use of custom Paddle-compatible model, place the model (including pdparameters and all files alongside) in the
./model directory

All further steps are same as for default model inference (see previous section)

# Model training 
To perform default model training, which can significantly increase recognition result, place your training data folder to the ./train_data directory and 
replace the default TRAINING_DATA_PATH from PATHS.txt file in the ./configs folrder.

Originally training data can be downloaded by the link here: https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset?resource=download

Notice, that the training data labels must be specified in a txt. file as follows: ![image](https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/datasets/icdar_rec.png)


You also may want to edit ./configs/default_model.yaml configuration file to control traninig learning rate, batch size, optimizer parameters and etc.
All avaliable changes are described here... # TODO
Then you can activate ./scripts/train_model.sh script and wait for training to be completed

# Custom model training 
To perform your custom model training, you should edit #TODO  


