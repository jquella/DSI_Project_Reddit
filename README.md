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



For this exercise, I also decided to drop all categorical variables to be able to more easily run different linear regression types. Some features appeared numerical, but were in fact categorical (e.g. `Year Built`).

The next step was to explore *feature correlations*. I did this by making a heatmap of the correlation matrix of the remaining features:

<img src="https://i.imgur.com/3MP09gR.png">

While a couple of features look highly correlated, we will be using some regularization techniques to drop those later.

Lastly, I needed to find out if there were still *null values* sticking around, which there were. The most important feature, `Lot Frontage`, we imputed by `Lot Shape`, which seemed intuitive.

<img src="https://i.imgur.com/WgNTP4a.png">

<a id='model_data'></a>
### Model Data.

I ran four different Linear Regression model types: 
1. *Linear Regression:* default
2. *Ridge Regression:* penalizes wrong guesses, but will not zero out coefficients.
3. *Lasso Regression:* penalizes wrong guesses, but *may* drop coefficients.
4. *Elastic Net:* Balance between *Ridge* and *Lasso*.

<a id='evaluate_model'></a>
### Evaluate Model.

*Scores for all models*

 **Model** | **R^2 Score (Train)** | **R^2 Score (Test)** 
 --- |	--- | --- 
 Linear Regression 	|	0.726	| 0.675 
 Ridge Regression		|	0.773	| 0.700 
 Lasso Regression 		|	0.775	| 0.698 
 Elastic Net 			|	0.774	| 0.697 

These R^2 scores are not ideal. While being able to explain nearly 80% of the variance in house price seems good, it can likely be improved. This mainly seems to be in part due to a large outlier in the predictions:

<img src="https://i.imgur.com/7irAXIT.png">

In exploring the data further, it seems that this was likely due to a single outlier data point in one of the strongest features: `GrLivArea` - or above ground living area.
<img src="https://i.imgur.com/7wVnQJe.png">

The next iterations of models should perhaps account for outliers in some way to generate better predictions.

<a id='answer_question'></a>
### Answer Question.

**Q: Were we able to predict home prices?** 

**A:** Somewhat. The R^2 scores  were fairly low (70-80% range) compared to what I think is possible to achieve. That said, I *did* achieve the objective of following the data science workflow, improving EDA, plotting and modeling skills, and was able to predict prices within the expected range. I just wouldn't make an investment strategy out of it :)

**Next steps**
1. More feature selection and engineering work.
    - Involve more categorical variables
    - Have separate models for outliers
    - Try polynomial features on more data
    - Use `itertools.combinations` to try more variations of feature combos
2. Use `GridSearchCV` to more exhaustively tune hyper-parameters
3. Try different models!

[Jupyter notebook with full code available here](https://github.com/jquella/DSI_Project_Ames/blob/master/project-2_JQ_take2.ipynb)