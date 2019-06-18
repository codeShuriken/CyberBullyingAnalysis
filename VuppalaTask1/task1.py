#I'm using Python 3.6 and Pandas for this task.
import pandas as pd

"""
This function reads through each comment and labels each comment as 1 if there is at least one
negative word/comment and or content with intent to harm someone or others, otherwise label it as 0

@param data: The dataframe that holds all the data
@return Returns a list which consists of the labels of each comment

"""
def labelComments(data):
    badwords = []
    #Store all the bad words in a list
    for line in open("badwords.txt"):
        for word in line.split( ):
            badwords.append(word)
        
    #Make sure every bad word is in lower case    
    badwords = [x.lower() for x in badwords]
    
    #Store all the comments in a seperate list
    comments = data.iloc[:, 4]
    
    #A list to store the labels
    label = []
    
    #Go through each comment
    for comment in comments:
        #Assume that the comment is not bad
        bad = False
        #Check each word in the comment
        for word in comment.split(" "):
            #Add 1 in the list if its a bad word
            if word.lower() in badwords:
                label.append(1)
                bad = True
                break
        #If no bad word is present in a comment, then add 0
        if bad == False:
            label.append(0)
    return label

"""This function goes through each comment and identifies the role of each comment.
    
    @param data: The dataframe that holds all the data
    @return Returns a list which consists of the roles of each comment
"""
def  roleClassifier(data):
    #A list to hold the roles
    roles = []
    labels = data.iloc[:, 5]
    for label in labels:
        if (label == 1):
            roles.append('bully')
        else:
            roles.append('other')
    return roles

def main():
    #Load the csv files data into a dataframe
    data = pd.read_csv('dataset_5.csv', sep=',', header=0)
    
    #print(data.values)
    #print(data.shape)
    
    #Label each comment
    data.iloc[:, 5] = labelComments(data)
    
    #Identify the role of each comment
    data.iloc[:, 6] = roleClassifier(data)
    
    #Add the labels to the csv file
    data.to_csv('dataset_5.csv', sep=',', encoding='utf-8', index=False)
    
if __name__ == '__main__':
    main()