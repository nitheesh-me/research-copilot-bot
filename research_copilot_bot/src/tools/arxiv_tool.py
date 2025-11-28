import arxiv
import os
from moya.tools.tool import Tool

def search_and_download_papers(query: str, max_results: int = 5, download_dir: str = "./pdfs"):
    """
    Search Arxiv for papers and download them.

    :param query: Search query.
    :param max_results: Maximum number of papers to download.
    :param download_dir: Directory to save PDFs.
    :return: List of downloaded file paths.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    downloaded_files = []
    for result in client.results(search):
        filename = f"{result.get_short_id()}.pdf"
        filepath = os.path.join(download_dir, filename)
        if not os.path.exists(filepath):
            result.download_pdf(dirpath=download_dir, filename=filename)
        downloaded_files.append(filepath)

    return downloaded_files

arxiv_tool = Tool(
    name="search_and_download_papers",
    description="Search Arxiv for papers and download them.",
    function=search_and_download_papers
)
