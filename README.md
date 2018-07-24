<img src="https://i.imgur.com/vIzKvgl.png">

# Predicting and Analyzing Top Comments from Reddit

_Jamie Quella - 6/4/18_

**Goal:** The goal of this project was to practice two major skills:
1. Collecting data by via an API
2. Building a binary predictor on text data.

We will do so by scraping or pulling Reddit comments on 'hot' threads and building a classification model that predicts whether or not a given Reddit post will have above or below the _median_ number of comments.


**Process:** 
1. [Define Question.](#define_question)
2. [Gather Data.](#gather_data)
3. [Explore Data.](#explore_data)
4. [Model Data.](#model_data)
5. [Evaluate Model.](#evaluate_model)
6. [Answer Question.](#answer_question)

<a id='define_question'></a>
### Define the Question.
_What characteristics of a comment on the NBA Reddit are most predictive of the overall interaction (as measured by comment score)?_


<a id='gather_data'></a>
### Gather Data.
Method for acquiring the data will be scraping the 'hot' threads as listed on the [NBA subreddit](https://www.reddit.com/r/nba/). 

To do so, I used [PRAW: THe Python Reddit API Wrapper](https://praw.readthedocs.io/en/latest/). PRAW allows anyone with Reddit Developer credentials to access the API and pull top posts, comment threads, and other information, compared with using the `requests` library, which must access the raw JSON for each URL.

**Data Collection Process**
1. Top 675 'hot' posts were pulled from the NBA subreddit as of 5/30/18.
2. All comments (~180K total) were pulled from each submitted post, based on `post_id`.
3. Looped through each comment thread to pull the following information from each comment:
	- `post_id`: identifier for loop
	- `gilded_status`: whether a comment was given "Reddit gold"
	- `author_team_flair`: indicator of commenter's favorite team
	- `top_comment`: whether it was a top-level comment or a threaded reply
	- `comment_score`: final upvote score

Due to size of data being pulled, the process needed to be chunked into 4 separate comment pulls and mega-concatenated at the end.


<a id='explore_data'></a>
### Explore Data.

Example data, once in dataframe: 

<img src="https://i.imgur.com/FGHXaeP.png">

#### Finding and Filling Null Values
First I took a look at all of the columns to check out the null values in the dataset, specifically trying to find columns which were:
	- Overwhelmingly null, and should be dropped, or
	- Partially null, and might have values imputed

<img src="https://i.imgur.com/xqUdBz4.png">

I soon realized that those null values for `flair_text` came from two places:
	- Deleted comments
	- Commenters who had not chosen a team flair

Obviously I wanted to drop the deleted comments, as there was no comment text to analyze. Not having a team flair in itself is a choice, so I filled those values with 'no_flair'. 
```
mask = df[df['comment_text'] == '[deleted]']
df = df.drop(mask.index)
df.flair_text = df.flair_text.fillna('no_flair')
```
Now we are ready to move onto the next step!

#### Standardizing Author Team Flair

**Author Flair was incredibly varied**
- Two different types: player flair, and team flair
- >1K unique player flairs, 117 team flairs
- *Player flair:* `[CLE] Kendrick Perkins`
- *Team flair:* `Cavaliers`

**So how do we reduce to an analyzable sample?**
- Regex word boundary matching on Team Flair
	- e.g. "Cavaliers bandwagon" put through this process:
	```
	long_flair['flair_text'].map(lambda x: re.findall('^\w+', x)[0].lower()).unique()
	```
	returns "cavaliers"
- Map function slicing on Player Flair
	- Similarly, "[CLE] Kendrick Perkins"
	```
	short_flair = df[df.flair_text.str.contains('\[')]['flair_text'].map(lambda x: x[1:4])
	```
	returns "CLE"

Lastly, I used a dictionary mapping to clean all 30 team flairs (+none, +other) to a standardized list:

<img src="https://i.imgur.com/hseqA9M.png">


#### Exploring Data Further.

I explored the mean comment score by author team flair to get a better sense of any prevailing trends.

<img src="https://i.imgur.com/724mGXb.png">

**Key Takeaways**
- Four conference Finals game and post-game threads contained > 60% of all comments 
- Many players complained of referees “rigging” WCF Game 7 in GSW’s favor
- People on the internet don’t like James Harden
- Grizzlies poor score likely because of variability due to low # of total comments 
- People felt bad for the Raptors, and a lot of those comments were self-deprecating jokes


Next I looked at *total* comments by team flair, to see if there was any correlation between volume of comments and score (e.g., most popular team flairs get downvoted more).

<img src="https://i.imgur.com/jTKwV57.png">

A-ha! That turned out to be mainly true for Golden State and Houston. I think Cleveland escaped the downvote hate because it was already seen as an underdog against whichever Western Conference team it would play in the Finals.

<a id='model_data'></a>
### Model Data.

*Feature Selection and Engineering*
- Features chosen: All binary features + author flair + comment text
- Process:
	- Dummieed variables for author flair
	- CountVectorized comment text data columns
	- Combined all columns back together

*Model selection*
- Due to computational constraints, only 50% of the data was sampled from non-game thread comments -- around 41K total
- RandomForestClassifier was chosen due to large number of text feature columns (approx. 21K) and feature weight given



<a id='evaluate_model'></a>
### Evaluate Model.

*Model evaluation*  

**Cross-validated accuracy score**
Final mean score: 55%    :(

This R^2 score is...not good. Some thoughts on why:
- Was not able to use the full dataset
- Text is a _much_ more variable feature and is harder to predict
	- Maybe TF-IDF or multiple n-grams would be more beneficial in illuminating top text
- Comment age (i.e., getting in early) may be a factor and was not explored at this time

**Feature Importance**
Our RandomForestClassifier (RF) returns _feature importances_, that is, which were the strongest predictors in determining whether a comment score would be high or low. See below for further detail.

<img src="https://i.imgur.com/ihCdYkn.png">

_What features were most important in generating a top comment?_
1. Being a top comment, rather than a reply, held the most predictive weight of any feature
2. Next was “other” or “no” flair
	- Commenters tend to upvote/downvote based on context of fandom rather than on content alone
3. Just mention “Lebron” by name


<a id='answer_question'></a>
### Answer Question.

**Q: Were we able to predict home prices?** 

**A:** Not very well. But we learned a lot. Some *Key Findings* and *Next Steps* below to illustrate what we learned, and how to improve.


**Key Findings**
1. Stake your own claim
	- Being a top comment, rather than a reply, were most important factor in determining comment score
2. Haters gonna hate
	- GSW and HOU fans got downvoted the most
3. Show your stripes
	- Comments with “no flair” scored poorly, so show your fandom! 


**Next steps**
1. More computing power!
	- Running on AWS or similar would enable faster analysis
	- GridSearchCV unavailable due to time constraints; hyper-parameter tuning could improve model 
2. 	Try other models!
	- LogisticRegression and SVM may improve outcomes 
	- N-gram text feature analysis


### Bonus! 
If you've made it this far, congratulations! I also generated some "top" comments via Markov chains, taken from some of the top-scoring comments I pulled. Here are some of my favorites:
- Am I supposed to be present to witness the inevitable.
- ﻿LeBron should just rent a place in the US and poster random dudes?
- Why lose in 7 when you can lose in 7 when you can lose in 7 when you can lose in 4 #efficiency
- Is it me or is harden pushing off like every loose ball. Nonstop hustle. Mad respect.



[Jupyter notebook with full code available here](https://github.com/jquella/DSI_Project_Reddit/blob/master/starter-code_jq-_v2.ipynb)