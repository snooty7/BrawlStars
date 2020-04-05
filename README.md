# BrawlStars
Convolutional Neural Network bot for navigating in Brawl Stars Android Game.

There will be a youtube tutorial soon i promise. :bowtie:

How does this script works?

This is an autonomous bot, which can navigate in Brawl Star map and shoots every 2 seconds. In order to use it correctly you need to 'teach' it how to move. So if you are a good player the results will be victories. In simple words the script will "watch" you playing the game. Then you will create an model using Python 3.7.4. This model will contain everything needed for the game. Remember that you are going to need a separate model for each map on Brawl Stars!!! The models are created only once then you can use them as much as you want.

This bot is created in base of Sentdex tutorials from this links:
 
https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/
 
https://github.com/sentdex/pygta5

To use this script you need to have instaled Python version 3.7.4!!!!!! I've try it with newer versions and there was a problem with pywinauto module.
There is a "requirements" file which can be installed with the following command : pip install -r requirements.txt
Not all of the dependencies in the file are required, but i don't thave time to clarify them.

You're going to need an android emulator for example i use NOX. You can download it from here: 

https://www.bignox.com/.

After you downloaded the files from github, open the main repository, start a command prompt in (or powershell), install the 'requirements' file. 
Separately start Nox emulator and move the window to your upper left corner. 
Nox emulator should be opened in window mode with size exactly 800x600px **IN YOUR UPPER LEFT CORNER OF YOUR SCREEN!!!**
If you are not sure about the size of the window you can always install an application that measure your screen dimentions. (A ruler for Windows).

Install Brawl Stars on Nox and then start the game. You can use your account from google play if you want to. Do not play yet. 
The python files which we are going to use are named with a digits at the begining. 
The first file is 1collect_data.py
You need to start it in the console(comand prompt) and immediately after that you need to start the game. The main purpose of this script is to collect your movements while you are playing the game. The script capture frames from your screen in 800x600px and keystrokes. Don't worry if the script starts and the game is still loading. We can clear the bad frames after that. The script will collect exactly 3000 frames with the following directions: straight, left, right, reverse, forward+left, forward+right, reverse+left, reverse+right and nokeys. 

After the game is over stop the script with key combination ctrl+c. Then you will see a new generated file **'training_data-1.npy'** which contains everything needed to continue.

Next step is to start from console **'2checkData.py'**. It will separate each frame depending of the move that you made while you played and then the frame will be saved in folder 'train'. For example all the 'left' movements goes to 'train\left' and  all the 'right' movements goes to 'train\right' and so on.
Then we need to examine each folder for 'bad'(damaged) frames and reduce the number of frames to be equal in each folder. This way the future model won't be overfitted.

After this start file **'3extract_features.py'**. This way we are going to extract the features we need in base of the inceptionv3 model. 

Next start file **'4train.py'**. This script will create a file **'model0.1.h5'** which is the model that we are going to use in the game. You need model for each map in Brawl Stars. At the time of writing there are 14 maps for Gem Grab mode in Brawl Stars. 
You can check them from here:

https://www.starlist.pro/maps/

The final step is to start in parallel **'5test_model.py'** and the game. After the game finished just stop the script ctrl+c. For the moment you need to start the script every time manually, but with some additional adjustment there is a way to make everything automatically.

The results as i tested are 7 wins from 10 games in Gem Grab mode.

Enjoy your play and modify as much as you want. :shipit:

P.S. There is a way to improve this script to be more presize in shooting and coordinate.





