module TFIDFVec

using TextAnalysis: StringDocument, Pipeline, StripPunctuation, LowerCaseTransformer, StripStopWords, Corpus, TFIDF, transform!
export tfidf_vectorize

function tfidf_vectorize(texts::Vector{String})
    # Create StringDocument objects from raw texts
    documents = StringDocument.(texts)

    # Preprocessing pipeline
    pipe = Pipeline(StripPunctuation(), LowerCaseTransformer(), StripStopWords())

    # Apply pipeline to documents
    for doc in documents
        pipe(doc)
    end

    # Create a corpus and apply TF-IDF
    corpus = Corpus(documents)
    model = TFIDF()
    transform!(model, corpus)

    # Convert sparse doc-term matrix to dense format
    return hcat([model.doc_term_matrix[i, :] for i in 1:length(documents)]...)'
end

end # module
