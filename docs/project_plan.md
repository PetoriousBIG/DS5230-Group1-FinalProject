Updated timelines on the sprint plan: 


**Sprint One: Create and execute process to scrape information**

Lu and Ehsan

Due: March 9 (if that is too soon we can move it)

As discussed in March 2 project meeting. The point of this is to gather comprehensive key words and topics to guide the process of creating parameters for future broader scraping of information.  The process at this point may scrape just keywords and executive summaries from Google Scholar. The .py file to execute the data scrape to be stored in src folder in the git repository. In sprint 3, we will run clustering analysis to identify the right parameters to organize existing data and new data.



**Sprint Two: Represent the data using unsupervised embeddings**

Everyone

Due: March 16

A major requirement here is to utilize Vector-database / RAG (Retrieval-Augmented Generation) approaches for storing and retrieving resources. You are expected to explore:

i. Use TF-IDF, Word2Vec, or Sentence-BERT to generate vector representations. 

ii. Students are encouraged to experiment with different embedding models, including popular LLM-based embedding generators.



**Sprint Three: Perform clustering analysis on the collected and embedded data**

Everyone

Due: March 23

As discussed in March 2 project meeting. Everyone picks a couple of algorithms to test on the data that's been collected and adds .py file to src folder and visualizations to the figs folder from that analysis. We will pick the best algorithm for clustering in our March 23 meeting.



**Sprint Four: Testing**

TBD

Due: March 31

Test the validity of the clusters (using the Google Scholar keywords?) and determine how to connect the core topic nodes



**Sprint Five: Process for augmenting data**

TBD

Due: April 7

Create process for automating incorporating additional information sources, which will require scraping the data and then identifying which cluster(s) this new information belongs to. Test validity of clusters again once new data has been ingested.




