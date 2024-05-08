
Our dataset is available at http://bnci-horizon-2020.eu/database/data-sets (dataset 007-2014). The data is in binary MATLAB (.mat) format and is 608.1 MB. To load the files into python, use the loadmat function from the included loadmat.py module. See the sample code for importing data for subject ‘S00’ below:


from loadmat import loadmat

data = loadmat('S00.mat')

Loading in the data for one subject yields a dictionary of size nine. The six keys which store trial information (as opposed to metadata) are:
chann     is a 1-dimensional array containing the labels of the 47 channels of data. The first 32 channels contain EEG data, EXG1 through EXG4 (4 channels) contain EOG data, EMG5 through EMG8 (4 channels) contain EMG data from finger movements, GSR1 contains galvanic skin response data, Plet containins blood pressure/heart rate data, Resp containins respiration data, and Temp containins temperature data.

I     is an 6 x n array of float values, where n is the number of samples collected for each study participant (24,5760). This array contains time values for each sample as well as information about experimental blocks, experimental conditions, and emotional self-report data. The labels describing the rows of this array can be found in id_lab.
id_lab     is a 1-dimensional array of size 6 containing the labels for the rows of array I.
markers     is an array of size 4 x 2 containing numerical markers and their respective meanings for each type of key press.
X     is an array float values of size 47 x n. The 47 rows correspond to the 47 labels in array chann. Time in seconds is given for each sample in the first row of array I.
Y     is a 1-dimensional array of n float values corresponding to the markers below (Table 1), which detail experimental conditions and information related to gameplay.


Table 1: Marker Meaning
1		key press with left index finger
2		key press with right index finger
3		LOC key press with left index finger
4		LOC key press with right index finger
5		screen freeze
10		init level
11		next level
12		Pacman avatar died
20		start game
21		end game
22		start normal condition
23		end normal condition
24		start frustration condition
25		end frustration condition
26		start self assessment
27		end self assessment
28		start pause
29		end pause
100–109	valence response
110–119	arousal response
120–129	dominance response