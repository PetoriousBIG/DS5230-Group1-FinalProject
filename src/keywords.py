def load_keywords(keyword_filepath="txt/keywords.txt"):
    with open(keyword_filepath, "r") as file:
        keywords = [
            line.strip() for line in file if line.strip()
        ]  # Read and clean up lines
    return keywords


if __name__ == "__main__":
    print(type(load_keywords()))
    print(" ".join(load_keywords()))
