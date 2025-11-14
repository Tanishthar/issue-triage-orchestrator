import pytest
import requests

def test_readme_structure():
    # Download the README file from the repository
    url = 'https://raw.githubusercontent.com/your-repo-name/main/README.md'
    response = requests.get(url)
    
    # Parse the README content using BeautifulSoup
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check if there are any duplicate or unnecessary link tags
    links = [tag for tag in soup.find_all('link')]
    assert len(links) == 2  # Assuming only two necessary link tags
    
    # Verify that all link attributes match expected standards
    for link in links:
        assert 'crossorigin' not in str(link)
        assert 'data-color-theme' not in str(link)

# This test will fail if the README file's HTML structure is incorrect