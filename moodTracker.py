import nltk  # natural language toolkit
import json
import os
from matplotlib import pyplot as plt
import string
from collections import Counter
import numpy as np
# sentiment is mood
# #polar words are non-neutral words (so positive or negative)


def getRespData():
    """
    The function sorts the lines of training strings from multiple csv files of data (mostly imported) into different lists based on their sentiment (4 is positive, 2 is neutral, 0 is negative) to improve the accuracy of the program's sentiment predictions.

    allResps = complete list of (response, sentiment) tuples from sample data

    files = list of file names from which I import string-sentiment data

    text = the polarized text from a line in the data file

    dataFile = current file with data being imported

    Note: no parameters, so no doctest
    """

    allResps = []
    files = [
        # http://help.sentiment140.com/for-students
        "stanfordSentiment140TweetData.csv",

        # https://old.datahub.io/dataset/twitter-sentiment-analysis/resource/091d6b4b-22e9-4a64-85c4-bdc8028183ac
        "dataHubTweetsFull.csv",

        # # crashes the program b/c it's too much data or takes too long to load; "MemoryError  Exception: No Description": https://www.kaggle.com/crowdflower/twitter-airline-sentiment/version/2
        # "airlineTweets2.csv"
    ]
    # custom data 20 times to outweigh specific words' tone in other files b/c some obvious words were being misclassified (bug cause currently unknown):
    for i in range(20):
        files.append("customData.csv")

    for i in range(len(files)):
        with open(files[i], "r", encoding="utf8") as dataFile:
            # w/o utf8 encoding some characters, are undefined
            for line in dataFile:
                line = line.strip().split(",", 1)
                try:
                    text = line[1].strip('"').lower()
                    if line[0] == '4':
                        allResps.append((text, "positive"))
                    elif line[0] == '2':
                        allResps.append((text, "neutral"))
                    elif line[0] == '0':
                        allResps.append((text, "negative"))
                except IndexError:
                    pass
        dataFile.close()

    return allResps


def initClassifier(allResps, ignoreWordsList):
    """
    The function first matches the filtered, possibly polar words (with usernames, punctuation, and unhelpful words removed) to their sentiment, then it prepares the sentiment classifier based on the training response examples to determine if a new given text response's sentiment is overall positive, neutral, or negative.

    Iterators: i, word

    allResps = complete list of (response, sentiment) tuples from sample data

    ignoreWordsList = list of common, neutral (or very context-dependent) words that are most likely accompanied by polar words (I don't want too many neutral words to overload an actually polar statement.)

    resps = list of (list of individual polar words, sentiment) tuples

    filteredWords = changing list of filtered words from a string (at least three characters long, not in the ignoreWordsList not a username (@), and not a url)

    sentimentTrainingList = list of every sentiment (for tracking num of responses)

    wordsTrainingList = list of all filteredWords lists

    sentiment = sample sentiment string from a tuple in allResps

    text = sample response string from a tuple in allResps

    textWordsList = list of text split into individual words

    sortedWords = list of filtered words ordered by decreasing frequency

    trainingData = list of labeled feature sets [a list of tuples ({dict of each resp's extracted features}, resp sentiment string)]

    classifier = object that maps each feature to the probability of it having a positive or negative sentiment

    Note: not doing a doc test run because the classifier doesn't return anything I can easily type
    """

    resps = []
    filteredWords = []
    sentimentTrainingList = []
    wordsTrainingList = []

    # filter and organize words from sample data:
    for i in range(len(allResps)):
        filteredWords = []
        sentiment = allResps[i][1]
        text = allResps[i][0]  # throws tuple error if I don't make new var
        textWordsList = text.split()

        # remove usernames:
        wordsToRemove = []
        for word in textWordsList:
            if word[0] == "@":
                wordsToRemove.append(word)
        textWordsList = [
            el for el in textWordsList if el not in wordsToRemove
        ]

        # join, remove punctuation (as str, not str-split list), re-split:
        text = ' '.join(textWordsList)
        text = text.translate(str.maketrans('', '', string.punctuation))
        textWordsList = text.split()

        # filter other unhelpful words:
        for word in textWordsList:
            if len(word) >= 3 and (word not in ignoreWordsList) and \
                    word[0:4] != "http":
                filteredWords.append(word)

        resps.append((filteredWords, sentiment))
        sentimentTrainingList.append(sentiment)
        wordsTrainingList.append((filteredWords))
    sortedWords = getWordFeatures(getRespsWords(resps))

    # compile training pairs (response, sentiment) and prepare classifier:
    trainingData = [
        (
            extractFeatures(sortedWords, wordsTrainingList[example]),
            sentimentTrainingList[example]
        ) for example in range(len(sentimentTrainingList))
    ]
    classifier = nltk.NaiveBayesClassifier.train(trainingData)
    return sortedWords, classifier


def classifyResp(userResp, ignoreWordsList, sortedWords, classifier):
    """
    This function removes punctuation and unhelpful words from the user's response, classifies the overall sentiment of the response, prints it, and returns the dictionary of mood data in the form {userResp, mood}.

    Iterators: word, el

    userResp = user's response about their day and feelings

    ignoreWordsList = list of common, neutral (or very context-dependent) words that are most likely accompanied by polar words (I don't want too many neutral words to overload an actually polar statement.)

    sortedWords = list of filtered words ordered by decreasing frequency

    extractedFeatures = dictionary with booleons for whether or not a sorted word is in the userResp

    classifier = object that maps each feature to the probability of it having a positive or negative sentiment

    mood = classified (predicted) sentiment for userResp

    moodDataDict = dictionary of {userResp string, mood string}

    Note: not doing a doc test run because the classifier isn't an object I can type
    """

    # remove punctuation from string and split string into list:
    userRespEdited = userResp.translate(
        str.maketrans('', '', string.punctuation)
    )

    # remove short/unhelpful words:
    userRespEdited = userRespEdited.split()
    wordsToRemove = []
    for word in userRespEdited:
        if len(word) < 3 or word in ignoreWordsList:
            wordsToRemove.append(word)
    userRespEdited = [el for el in userRespEdited if el not in wordsToRemove]

    extractedFeatures = extractFeatures(sortedWords, userRespEdited)

    mood = classifier.classify(extractedFeatures)
    print(f'\nYour response was overall {mood}.')
    respond(userRespEdited, mood)
    moodDataDict = {userResp: mood}
    return moodDataDict


def getRespsWords(resps):
    """
    The function turns a list of tuples containing pairs: [([filtered words], sentiment)] into a list of the filtered words only to prepare it for the classifier.

    Iterators: words, sentiment (unused)

    resps = full list of (list of polar words in a string, sentiment)

    words = list of filtered words

    sigWords = list of all individual filtered words from resps

    >>> getRespsWords([(['love', 'best', 'friend'], 'positive')])
    ['love', 'best', 'friend']
    """
    sigWords = []
    for (words, sentiment) in resps:
        sigWords.extend(words)
    return sigWords


def getWordFeatures(sigWords):
    """
    The function reorders the list of filtered words by decreasing frequency to prepare it for the classifier.

    Iterators: word, freq (unused)

    sigWords = list of all filtered individual words from the string

    sigWordsAndFreq = list of tuples in the form (filtered word, frequency) ordered by decreasing frequency

    sortedWords = list of filtered words ordered by decreasing frequency

    >>> getWordFeatures(['sky', 'like', 'pie', 'like', 'like', 'pie'])
    ['like', 'pie', 'sky']
    """
    sortedWords = []
    sigWordsAndFreq = Counter(sigWords).most_common()
    for (word, freq) in sigWordsAndFreq:
        sortedWords.append(word)
    return sortedWords


def extractFeatures(sortedWords, words):
    """
    The function is a feature extractor that compares words in a response (from training or from user) to the words in a list of possible polar words so that unused words can be ignored and, depending on when the function is called, the trainig data can be prepared or the user's response can be tested against the training data.

    Iterators: word

    sortedWords = dictionary keys list of filtered words ordered by decreasing frequency

    words = all filtered response words split into a list

    features = dictionary of {'contains(polar word)': boolean whether or not the user input string contains the polar word}

    >>> extractFeatures(['like', 'pie', 'fly', 'sky'], ['I', 'like', 'pie'])
    {'contains(like)': True, 'contains(pie)': True, 'contains(fly)': False, 'contains(sky)': False}

    """
    wordSet = set(tuple(words))
    features = {}
    for word in sortedWords:
        features[f'contains({word})'] = (word in wordSet)
    return features


def respond(userRespWords, mood):
    """
    The function searches for key words in the user's response in order to respond to the user's entry when it is helpful for certain situations.

    userRespWords = list of useful individual words filtered from the user's response

    mood = predicted sentiment of user's response

    sleepTipKW = list of key words to trigger sleep tips

    selfCareKW = list of key words to trigger affirmation of a positive experience

    sickKW = list of key words to trigger suggestions regarding sickness

    >>> respond(["tired","not","sleep"], "negative")
    The average adult needs to sleep for 7-9 hours. If you find yourself laying awake at night, there are several things you can do to help you sleep.
    Eat foods with melatonin around bedtime, such as cherries.
    To relax, you can try the 4-7-8 breathing technique: breathe in for 4 seconds, hold your breath for 7 seconds, and breathe out for 8 seconds.
    You can also flex all of your muscles, starting at your feet and gradually working your way up to your head, then gradually relax them, again, starting from your feet and working your way up to your head.
    """

    sleepTipKW = ["sleepy", "sleep", "drowsy"]
    selfCareKW = ["holiday", "vacation", "relax", "party", "festival"]
    sickKW = ["sick", "illness", "unwell", "sickness", "fever"]

    if mood == "negative" and (set(userRespWords) & set(sleepTipKW)):
        print(
            "The average adult needs to sleep for 7-9 hours. "
            "If you find yourself laying awake at night, there are several things you can do to help you sleep."
            "\nEat foods with melatonin around bedtime, such as cherries."
            "\nTo relax, you can try the 4-7-8 breathing technique: "
            "breathe in for 4 seconds, hold your breath for 7 seconds, and breathe out for 8 seconds."
            "\nYou can also flex all of your muscles, starting at your feet and gradually working your way up to your head, "
            "then gradually relax them, again, starting from your feet and working your way up to your head."
        )
    elif mood == "positive" and (set(userRespWords) & set(selfCareKW)):
        print(
            "Yay! Life is too short to not enjoy yourself. "
            "Always remember that you ARE worth it."
        )
    elif mood == "negative" and (set(userRespWords) & set(sickKW)):
        print(
            "If you need to take a sick day tomorrow to rest, don't be afraid to do it. "
            "It will keep the sickness from spreading to others and will allow your body to fight it, so it's a win-win. "
            "\nAnd if you worry that you are seriously ill, then visit the doctor, rather than trying to diagnose and treat yourself."
        )
    elif mood == "negative":
        print(
            "Oof That's rough."
            "Do your best to make tomorrow a better day :)"
        )
    elif mood == "positive":
        print("Yay! Happy days are the best days!")


def storeMood(entry):
    """
    The function adds dictionaries in the form {resp: mood} to the json file.

    dataFile = file of json mood data to read

    entry = new user entry (response)

    moodData = updated json mood data to store

    oldMoodData = json mood data (user entries) already stored

    Note: nothing is printed because it just writes to a file, so no doctest
    """
    with open("moodTrackerData.json", "r+") as dataFile:
        if os.path.getsize("moodTrackerData.json") == 0:  # file empty
            moodData = entry
        else:  # file occupied
            oldMoodData = json.load(dataFile)
            moodData = {**oldMoodData, **entry}  # consolidates duplicate pairs
        dataFile.seek(0)
        dataFile.truncate()
        dataFile.write(json.dumps(moodData))


def graphSentiments():
    """
    The function graphs the sentiment of entries (negative, neutral, or positive) vs the entry number so that the user can see the general trend of their mood over the course of entries. Doesn't graph and instead prompts the user to make another entry if there are less than 2 entries stored.

    Iterators: entry, mood, e

    dataFile = read file of json mood data

    entries = dictionary of user entries stored   

    entriesCount = list of integers from 1 to the number of entries + 1

    sentimentList = list of sentiments (used for y-axis tick labels)

    Note: no parameters nor typeable output, so no doctest
    """

    try:
        with open("moodTrackerData.json", "r+") as dataFile:
            entries = json.load(dataFile)
            entriesCount = [entry for entry in range(1, len(entries)+1)]
            sentimentList = [mood for mood in entries.values()]
            if len(entriesCount) == 1:  # graph would just be a dot, not useful
                raise ValueError

        plt.plot(entriesCount, sentimentList, 'bo-')  # x, y, marker
        plt.yticks(sentimentList, sentimentList)
        if len(entriesCount) < 6:  # scaled if < 6 , intervals of 1 if >= 6
            plt.xticks(np.arange(min(entriesCount), max(entriesCount)+1, 1.0))
        plt.ylabel('Sentiment')
        plt.xlabel('Entry Number')
        plt.title('Sentiment vs Entry Number Graph')
        plt.show()

    except (json.decoder.JSONDecodeError, ValueError) as e:
        print(
            "Sorry. There's not enough data to graph. Make a new entry first."
        )


def deleteEntries():
    """
    This simple function overwrites the json file that stores the entry data in order to delete all entries

    Note: super simple and no parameters, so no doctest
    """
    open("moodTrackerData.json", "w").close()
    print("You have deleted all of your entries.")


def main():
    """
    The function serves as the program's main menus and loops in asking the user to chose between a variety of actions (make a new entry, graph your mood data, delete all entries, or quit the program) and calling the function for that option until the user quits.

    ignoreWordsList = list of common, neutral (or very context-dependent) words that are most likely accompanied by polarized words (I don't want too many neutral words to overload an actually polarized statement.)

    yesList = list of possible affirmative responses to the repeat program question used to determine that the user wants to perform further actions

    repeat = whether the user wants to repeat the program or not, used to facilitate loop

    allResps = complete list of (response, sentiment) tuples from sample data

    userResp = user's response about their day

    sortedWords = list of filtered words ordered by decreasing frequency

    classifier = object that maps each feature to the probability of it having a positive or negative sentiment

    Note: no parameters, so no doctest
    """
    ignoreWordsList = [  # must be lowercase and w/o punctuation
        "feel", "feeling", "need", "want", "and", "then", "day", "night", "had", "has", "have", "make", "makes", "made", "was", "are", "were", "will", "afternoon", "evening", "morning", "get", "got", "receive", "received", "the", "went", "its", "his", "her", "their", "our", "they", "them", "this", "that", "im", "mr", "mrs", "ms", "for", "you", "with", "only", "essentially", "basically", "from", "but", "just", "also", "too", "out", "today", "tonight", "tomorrow", "about", "around", "watch", "watched", "see", "saw", "hear", "heard", "now", "currently", "all", "what", "who", "where", "when", "how", "why", "some", "lots", "very", "really", "much", "many", "someone", "something", "since", "because", "which", "there", "did", "more", "less", "ate", "eat", "drink", "drank"
    ]

    yesList = [
        'yes', 'yeah', 'sure', 'okay', 'ok', 'why not', 'yeet', 'yep', 'yup', 'si', 'affirmative', 'of course', 'always'
    ]
    repeat = 'yes'
    print(
        "Welcome to KT's Mood Tracker!\n"
        "Please wait while the sentiment classifier initializes."
    )
    allResps = getRespData()
    sortedWords, classifier = initClassifier(allResps, ignoreWordsList)

    while repeat == "yes":
        userResp = input(
            "\nTo make a new entry, type 'entry',\n"
            "to make a graph of your mood over the course of your entries so far, type 'graph',\n"
            "to delete all of your entries, type 'delete',\n"
            "or to exit the program, type 'quit'.\n"
        ).strip().lower()

        if userResp == "entry":
            userResp = input(
                "\nPlease describe your day and how you feel about it."
                "\nStick to one general tone in your entry, please, or "
                "this version of the mood tracker will get confused. "
                "\nAlso, the mood tracker does not understand sarcasm.\n"
            ).lower()
            storeMood(classifyResp(
                userResp, ignoreWordsList, sortedWords, classifier
            ))

            repeat = input(
                "\nDo you want to do another action?\n"
            ).lower()
        elif userResp == "graph":
            graphSentiments()
        elif userResp == "delete":
            deleteEntries()
        elif userResp == "quit":
            repeat = "no"
        else:
            print("Error! Try again.")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
