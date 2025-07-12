using CSV
using DataFrames
using Statistics

# Load dataset
function load_dataset(file_path)
    println("Loading dataset from: $file_path")
    df = CSV.read(file_path, DataFrame; ignoreemptyrows=true)
    println("Loaded $(nrow(df)) rows and $(ncol(df)) columns.")
    println("Columns: ", names(df))
    
    # Show first few rows
    println("\nFirst 3 rows of the dataset:")
    show(first(df, 3), allrows=true, allcols=true)

    return df
end

# Preprocess tweet text
function preprocess_text(df::DataFrame)
    println("\nStarting text preprocessing...")

    # Missing value summary
    for col in [:tweet_text, :cyberbullying_type]
        missing_count = count(ismissing, df[!, col])
        println("Missing in $col: $missing_count")
    end

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

    println("Example before cleaning:")
    println(df.tweet_text[findfirst(!ismissing, df.tweet_text)])

    df.tweet_text = preprocess_single_text.(df.tweet_text)
    df = dropmissing(df, :tweet_text)

    println("Example after cleaning:")
    println(df.tweet_text[1])

    # Convert multi-class to binary classification
    println("\nOriginal cyberbullying_type value counts:")
    println(combine(groupby(df, :cyberbullying_type), nrow => :Count))

    df.label = ifelse.(df.cyberbullying_type .== "not_cyberbullying", 0, 1)

    println("Binary label distribution (0 = not_cyberbullying, 1 = cyberbullying):")
    println(combine(groupby(df, :label), nrow => :Count))

    # Remove empty or missing rows
    df = filter(row -> !ismissing(row.tweet_text) && !isempty(row.tweet_text), df)
    println("After preprocessing: $(nrow(df)) valid rows retained.")

    return df
end

# Vectorize text using hash-based vectorizer
function vectorize_text(df::DataFrame, dim=3000)
    println("\nVectorizing text using hash-based method with dimension = $dim")
    mat = zeros(Float32, nrow(df), dim)
    for (i, text) in enumerate(df.tweet_text)
        words = split(text)
        for word in words
            h = hash(word) % dim + 1
            mat[i, h] += 1
        end
    end

    y = convert(Array{Float32}, df.label)

    println("Vectorization complete.")
    println("Feature matrix shape: ($(size(mat, 1)), $(size(mat, 2)))")
    println("Label vector length: ", length(y))

    return mat, y
end
