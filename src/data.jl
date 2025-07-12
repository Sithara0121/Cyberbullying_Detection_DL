# src/data.jl
using CSV
using DataFrames

# Load dataset
function load_dataset(file_path)
    df = CSV.read(file_path, DataFrame; ignoreemptyrows=true)
    return df
end

# Preprocess tweet text
function preprocess_text(df::DataFrame)
    function preprocess_single_text(text)
        try
            text = lowercase(String(text))
            text = replace(text, r"http\S+" => "")
            text = replace(text, r"[^a-z\s]" => "")
            text = replace(text, r"\s+" => " ")
            return strip(text)
        catch e
            return ""
        end
    end
    df.tweet_text = preprocess_single_text.(df.tweet_text)
    df = dropmissing(df, :tweet_text)

    # Convert multi-class to binary classification
    df.label = ifelse.(df.cyberbullying_type .== "not_cyberbullying", 0, 1)
    return filter(row -> !ismissing(row.tweet_text) && !isempty(row.tweet_text), df)
end

# Vectorize text using hash-based vectorizer
function vectorize_text(df::DataFrame, dim=3000)
    mat = zeros(Float32, nrow(df), dim)
    for (i, text) in enumerate(df.tweet_text)
        words = split(text)
        for word in words
            h = hash(word) % dim + 1
            mat[i, h] += 1
        end
    end
    y = convert(Array{Float32}, df.label)
    return mat, y
end
