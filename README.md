# PyTorch Summer 2020 Hackathon

This project was built for the PyTorch Summer 2020 Hackathon. It has a few pieces that are still a work in progress, but the general outline is below. The submission can be viewed [here](https://devpost.com/software/math-notetaker).

## Documentation Similarity

I scraped the documentation for machine learning packages like Scipy, Numpy, Caret, and Sklearn and built embedding vectors for each section using a doc2vec / paragraph vectors model. I then matched the top 5 most similar vectors to each section to enable recommendations between packages. A more constructed version of this project could serve as an easy way to onboard to new general machine learing packages and find relevant resources to flatten the learning curve. 

The recommendations and t-SNE projections of the vectors can be viewed on [this dash app](http://spencerbraun.pythonanywhere.com/), hosted on pythonanywhere. 

## arXiv Article recommendations

I started on a second direct, embedding arXiv abstracts for machine learning papers using a BERT transformer. This work is ongoing but could similarly serve to relate packages to relevant papers that provide the theory behind the algorithms. 


## Future Directions

I started building a CNN model as another way to create document embeddings. While it wasn't finished in time for the competition, language CNNs are an interesting way to maintain structure in embeddings.