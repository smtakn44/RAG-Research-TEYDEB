from langchain_community.tools import DuckDuckGoSearchResults

def web_search_agent(query):
    # DuckDuckGo arama sonuçlarını almak için araç
    search = DuckDuckGoSearchResults(max_results=10)

    # Arama yap
    results = search.invoke(query)

    # Sonuçları parçalarına ayır
    result_parts = results.strip('[]').split('], [')

    # Snippet'leri toplamak için bir liste oluştur
    snippets = []
    snippets_str = ""

    # Her bir sonucu işle
    for result_part in result_parts:
        # Her bir parçayı temizle ve ayrıştır
        result_part = result_part.strip()
        if result_part:
            # Snippet bilgisini ayır
            snippet = result_part.split(', title: ')[0].replace('snippet: ', '').strip()
            snippets.append(snippet)

    # Snippet'leri string'e ekle
    for index, snippet in enumerate(snippets):
        snippets_str += f"Web Snippet {index + 1}: {snippet}\n"

    return snippets_str

if __name__ == "__main__":
    response = web_search_agent("Tübitak kobiler için ne gibi destekleri vardır?")
    print(response)
