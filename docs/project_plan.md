Updated timelines on the sprint plan: 


**Sprint One: Create and execute process to scrape information**

Lu and Ehsan

Due: March 9 (if that is too soon we can move it)

As discussed in March 2 project meeting. The point of this is to gather comprehensive key words and topics to guide the process of creating parameters for future broader scraping of information.  The process at this point may scrape just keywords and executive summaries from Google Scholar. The .py file to execute the data scrape to be stored in src folder in the git repository. In sprint 3, we will run clustering analysis to identify the right parameters to organize existing data and new data.

**Spring 1.5: EDA, data preprocessing, and dimensionality reduction

NEED A VOLUNTEER

Deliverable from the EDA leaders: Explain what you discovered in the data, what data preprocessing and dimienstionality reduction was performed

**Sprint Two: Represent the data using unsupervised embeddings**

EVERYONE NEEDS TO PICK AN APPROACH AND STAY IN THEIR LANE

Due: March 16

A major requirement here is to utilize Vector-database / RAG (Retrieval-Augmented Generation) approaches for storing and retrieving resources. You are expected to explore:

i. Use TF-IDF, Word2Vec, or Sentence-BERT to generate vector representations. 

ii. Students are encouraged to experiment with different embedding models, including popular LLM-based embedding generators.

Deliverable from the whole team: A comparative study to determine which embedding is the best one for our data: As part of the process of testing your approach, note the strengths and weakness for your approach for our data

**Sprint Three: Perform clustering analysis on the collected and embedded data**

We will discuss which algorhithms to test once our embedding work is completed, hopefully by March 16, and then each of us will work on a different algorhithm to which we have assigned ourselves

Due: March 31

Once we have decided which algorhitms to test, each of us will pick a different one on which to focus and stay in our lane. Evaluate the model results as well, then store .py file to src folder and visualizations to the figs folder from that analysis. We will pick the best algorithm for clustering in our March 31 meeting.

Deliverable from the whole team:  Explain why we picked the algorhithm we picked as best

**Sprint Four: Mind Mapping**

TBD

Due: April 7

How do we mind map the different clusters so that there is a progression from one to the next?  


**Sprint Five: Process for augmenting data**

TBD

Due: April 14

Create process for automating incorporating additional information sources, which will require scraping the data and then identifying which cluster(s) this new information belongs to. Test validity of clusters again once new data has been ingested.


Three minimum outcomes for the project:
* Best dimensionality reductions and preprocessing activity 
* Best vectorization embedding process 
* Best algorithm(s) for clustering 

