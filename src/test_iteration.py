from test import iteration
from arxiv import main


def count_lines(filename):
    try:
        with open(filename, "r") as file:
            line_count = len(file.readlines())
        return line_count
    except FileNotFoundError:
        return None


def read_line(file_path, line_number):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            if 1 <= line_number <= len(lines):
                return lines[line_number - 1].rstrip("\n")
            else:
                return None
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    temp_file_path = "temp/temp.txt"
    keywords_to_search = "txt/output_keywords.txt"
    num_clusters = count_lines(keywords_to_search)
    iter = 1
    print(f"This is iteration-{iter} after the initial clustering.")
    print(f"Number of clusters from previous iteration is: {num_clusters}")

    for cluster in range(num_clusters):
        print(f"Fetching keywords from cluster {cluster} of the previous iteration.")
        text_content = read_line(
            keywords_to_search, cluster + 1
        )  # .replace(", ", "\n")
        with open(temp_file_path, "w") as file:
            file.write(text_content)

        main(
            input_txt=temp_file_path,
            output_csv=f"data/Arxiv_Resources_i{iter}_cluster{cluster}.csv",
        )

        iteration(
            input_keyword=f"data/Arxiv_Resources_i{iter}_cluster{cluster}.csv",
            output_keyword=f"txt/output_Arxiv_i{iter}_cluster{cluster}.txt",
            SBERT_pretrain="all-MiniLM-L6-v2",
            fig_dir=f"fig/iter{iter}_cluster{cluster}/",
        )
